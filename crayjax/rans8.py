from functools import partial, lru_cache

from collections import namedtuple
import numpy as np
from jax.scipy.stats import norm
import jax.numpy as jnp
from jax import lax, linear_transpose, tree_multimap, tree_map
from jax.util import safe_map
from jax import jit

from jax import lax


Codec = namedtuple('Codec', ['push', 'pop'])

head_prec, head_dtype = 32, 'uint32'
tail_prec, tail_dtype = 8,  'uint8'
head_min = 1 << head_prec - tail_prec
atleast_1d = lambda x: jnp.atleast_1d(x).astype(head_dtype)

_copy = jit(lambda x: (x + 1) - 1)

def base_message(shape, tail_capacity):
    return jnp.full(shape, head_min, head_dtype), empty_stack(tail_capacity)

def empty_stack(capacity):
    return jnp.array([capacity]), jnp.zeros(capacity, tail_dtype)

def stack_push(stack, idxs, arr):
    idxs, arr = jnp.ravel(idxs), jnp.ravel(arr)
    limit, data = stack
    return limit - idxs.sum(), lax.dynamic_update_slice(
        data, lax.sort_key_val(idxs, arr)[1], limit - arr.size)

def stack_pop(stack, idxs):
    idxs_flat = jnp.ravel(idxs)
    limit, data = stack
    unsorted = lax.sort_key_val(idxs_flat, jnp.arange(idxs.size))[1]
    limit = limit + idxs.sum()
    return (limit, data), jnp.reshape(
        lax.sort_key_val(unsorted,
                         lax.dynamic_slice(data, limit - idxs.size,
                                           idxs_flat.shape))[1], idxs.shape)

def stack_check(stack):
    limit, _ = stack
    assert limit >= 0

def stack_copy(stack):
    return map(_copy, stack)

@partial(jit, donate_argnums=(0,))
def push_core(m, starts, freqs, precs):
    head, tail = m
    idxs = head >> 2 * tail_prec >= freqs << head_prec - precs
    tail = stack_push(tail, idxs, head.astype(tail_dtype))
    head = jnp.where(idxs, head >> tail_prec, head)

    idxs = head >> 1 * tail_prec >= freqs << head_prec - precs
    tail = stack_push(tail, idxs, head.astype(tail_dtype))
    head = jnp.where(idxs, head >> tail_prec, head)

    idxs = head >= freqs << head_prec - precs
    tail = stack_push(tail, idxs, head.astype(tail_dtype))
    head = jnp.where(idxs, head >> tail_prec, head)

    head_div_freqs, head_mod_freqs = jnp.divmod(head, freqs)
    return (head_div_freqs << precs) + head_mod_freqs + starts, tail

@partial(jit, donate_argnums=(0,))
def pop_core(m, cfs, starts, freqs, precs):
    head_, tail = m
    head = freqs * (head_ >> precs) + cfs - starts
    for _ in range(3):
        idxs = head < head_min
        tail, new_head = stack_pop(tail, idxs)
        head = jnp.where(idxs, head << tail_prec | new_head, head)
    return head, tail

@jit
def peek_core(m, precisions):
    head, _ = m
    return head & ((1 << precisions) - 1)

def NonUniform(enc_fun, dec_fun, precisions):
    precisions = atleast_1d(precisions)
    def push(m, x):
        starts, freqs = map(atleast_1d, enc_fun(x))
        return push_core(m, starts, freqs, precisions)

    def pop(m):
        cfs = peek_core(m, precisions)
        x = dec_fun(cfs)
        starts, freqs = map(atleast_1d, enc_fun(x))
        return pop_core(m, cfs, starts, freqs, precisions), x
    return Codec(push, pop)

def message_copy(m):
    head, tail = m
    return _copy(head), stack_copy(tail)

def flatten(m):
    head, ([tail_limit,], tail_data) = m
    head = jnp.ravel(head)
    return jnp.concatenate([(head >> 3 * tail_prec).astype(tail_dtype),
                            (head >> 2 * tail_prec).astype(tail_dtype),
                            (head >> tail_prec).astype(tail_dtype),
                            head.astype(tail_dtype), tail_data[tail_limit:]])

def unflatten(arr, shape, tail_capacity):
    size = np.prod(shape)
    head_highest, head_high, head_low, head_lowest, tail = jnp.split(
        arr, [size, 2 * size, 3 * size, 4 * size])
    head = (head_highest.astype(head_dtype) << 3 * tail_prec
            | head_high.astype(head_dtype)  << 2 * tail_prec
            | head_low.astype(head_dtype)   << 1 * tail_prec
            | head_lowest.astype(head_dtype))
    tail_limit = tail_capacity - tail.size
    tail = (jnp.array([tail_limit]),
            jnp.concatenate([jnp.zeros(tail_limit, tail_dtype), tail]))
    return jnp.reshape(head, size), tail

def message_equal(message1, message2):
    return jnp.all(flatten(message1) == flatten(message2))

_uniform_enc_statfun = lambda s: (s, 1)
_uniform_dec_statfun = lambda cf: cf
def Uniform(precision):
    return NonUniform(_uniform_enc_statfun, _uniform_dec_statfun, precision)

def view_update(data, view_fun):
    item, view_transpose = view_fun(data), linear_transpose(view_fun, data)
    item_copy = tree_map(_copy, item)
    def update(new_item):
        diff, = view_transpose(tree_multimap(jnp.subtract, new_item, item_copy))
        return tree_multimap(jnp.add, data, diff)
    return item, update

def substack(codec, view_fun):
    def push(message, data, *args, **kwargs):
        head, tail = message
        subhead, update = view_update(head, view_fun)
        subhead, tail = codec.push((subhead, tail), data, *args, **kwargs)
        return update(subhead), tail
    def pop(message, *args, **kwargs):
        head, tail = message
        subhead, update = view_update(head, view_fun)
        (subhead, tail), data = codec.pop((subhead, tail), *args, **kwargs)
        return (update(subhead), tail), data
    return Codec(push, pop)

def _nearest_uint(arr):
    return jnp.uint32(jnp.ceil(arr - 0.5))

def _undiff(x):
    return jnp.concatenate([jnp.zeros_like(x, shape=(*x.shape[:-1], 1)),
                            jnp.cumsum(x, -1)], -1)

def CategoricalUnsafe(weights, prec):
    cumweights = _undiff(weights)
    cumfreqs = _nearest_uint((1 << prec) * (cumweights / cumweights[..., -1:]))
    def enc_statfun(x):
        lower = jnp.take_along_axis(cumfreqs, x[..., None], -1)[..., 0]
        upper = jnp.take_along_axis(cumfreqs, x[..., None] + 1, -1)[..., 0]
        return lower, upper - lower
    def dec_statfun(cf):
        # One could speed this up for large alphabets by
        #   (a) Using vectorized binary search
        #   (b) Using the alias method
        return jnp.argmin(cumfreqs <= cf[..., None], axis=-1) - 1
    return NonUniform(enc_statfun, dec_statfun, prec)

def Bernoulli(p, prec):
    p = jnp.clip(_nearest_uint(p * (1 << prec)), 1, (1 << prec) - 1)
    onemp = (1 << prec) - p
    enc_statfun = lambda x: (jnp.where(x, onemp, 0), jnp.where(x, p, onemp))
    dec_statfun = lambda cf: jnp.uint32(cf >= onemp)
    return NonUniform(enc_statfun, dec_statfun, prec)


@partial(jit, donate_argnums=(2,))
def categorical_push(weights, prec, message, x):
    return substack(CategoricalUnsafe(weights, prec),
                    lambda x: x[0]).push(message, x)

@partial(jit, donate_argnums=(2,))
def categorical_pop(weights, prec, message):
    return substack(CategoricalUnsafe(weights, prec),
                    lambda x: x[0]).pop(message)

@lru_cache
def std_gaussian_buckets(precision):
    return norm.ppf(jnp.linspace(0, 1, (1 << precision) + 1))

@lru_cache
def std_gaussian_centres(precision):
    return norm.ppf((jnp.arange(1 << precision) + 0.5) / (1 << precision))

def _gaussian_cdf(mean, stdd, prior_prec, post_prec):
    def cdf(idx):
        x = std_gaussian_buckets(prior_prec)[idx]
        return _nearest_uint(norm.cdf(x, mean, stdd) * (1 << post_prec))
    return cdf

def _gaussian_ppf(mean, stdd, prior_prec, post_prec):
    cdf = _gaussian_cdf(mean, stdd, prior_prec, post_prec)
    def ppf(cf):
        x = norm.ppf((cf + 0.5) / (1 << post_prec), mean, stdd)
        # Binary search is faster than using the actual gaussian cdf for the
        # precisions we typically use, however the cdf is O(1) whereas search
        # is O(precision), so for high precision cdf will be faster.
        idxs = jnp.uint32(jnp.digitize(x, std_gaussian_buckets(prior_prec)) - 1)
        # This loop works around an issue which is extremely rare when we use
        # float64 everywhere but is common if we work with float32: due to the
        # finite precision of floating point arithmetic, norm.[cdf,ppf] are not
        # perfectly inverse to each other.
        idxs = lax.while_loop(
            lambda idxs: ~jnp.all((cdf(idxs) <= cf) & (cf < cdf(idxs + 1))),
            lambda idxs: jnp.select(
                [cf < cdf(idxs), cf >= cdf(idxs + 1)],
                [idxs - 1,       idxs + 1           ], idxs),
            idxs)
        return idxs
    return ppf

def DiagGaussian_StdBins(mean, stdd, coding_prec, bin_prec):
    """
    Codec for data from a diagonal Gaussian with bins that have equal mass under
    a standard (0, I) Gaussian
    """
    def enc_statfun(x):
        lower = _gaussian_cdf(mean, stdd, bin_prec, coding_prec)(x)
        upper = _gaussian_cdf(mean, stdd, bin_prec, coding_prec)(x + 1)
        return lower, upper - lower
    dec_statfun = _gaussian_ppf(mean, stdd, bin_prec, coding_prec)
    return NonUniform(enc_statfun, dec_statfun, coding_prec)
