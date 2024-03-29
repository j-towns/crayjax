"""
Vectorized JAX implementation of rANS, by Jamie Townsend. Based on ideas from
'Interleaved entropy coders' by Fabian Giesen
(https://arxiv.org/abs/1402.3392), and the Craystack library
(https://github.com/j-towns/craystack).
"""
from functools import partial
from collections import namedtuple

import numpy as np

from jax.scipy.stats import norm
import jax.numpy as jnp
from jax import lax, linear_transpose, tree_multimap
from jax.util import safe_map


map = safe_map
atleast_1d = lambda x: jnp.atleast_1d(x).astype(head_dtype)


########################### Vectorized rANS core ##############################
head_prec, head_dtype = 32, 'uint32'
tail_prec, tail_dtype = 8,  'uint8'
head_min = 1 << head_prec - tail_prec

def base_message(size, tail_capacity):
    return jnp.full((size,), head_min, head_dtype), empty_stack(tail_capacity)

def empty_stack(capacity):
    return jnp.array([capacity]), jnp.zeros(capacity, tail_dtype)

def _write_selector(idxs):
    return jnp.where(idxs, jnp.cumsum(idxs), 0) - 1

def _read_selector(idxs):
    # Don't need to worry about overwriting, so can make this slightly faster
    # than the write selector.
    return jnp.cumsum(idxs) - 1

def stack_push(stack, idxs, arr):
    limit, data = stack
    limit = limit - idxs.sum()
    return limit, data.at[limit + _write_selector(idxs)].set(arr)

def stack_pop(stack, idxs):
    limit, data = stack
    return (limit + idxs.sum(), data), data[limit + _read_selector(idxs)]

def stack_check(stack):
    limit, data = stack
    capacity, = data.shape
    assert 0 <= limit <= capacity

def push(m, starts, freqs, precs):
    head, tail = m
    for i in reversed(range(3)):
        idxs = head >> i * tail_prec + head_prec - precs >= freqs
        tail = stack_push(tail, idxs, head.astype(tail_dtype))
        head = jnp.where(idxs, head >> tail_prec, head)
    head_div_freqs, head_mod_freqs = jnp.divmod(head, freqs)
    return (head_div_freqs << precs) | head_mod_freqs + starts, tail

def pop(m, cfs, starts, freqs, precs):
    head, tail = m
    head = freqs * (head >> precs) + cfs - starts
    for _ in range(3):
        idxs = head < head_min
        tail, new_head = stack_pop(tail, idxs)
        head = jnp.where(idxs, head << tail_prec | new_head, head)
    return head, tail

def peek(m, precisions):
    head, _ = m
    return head & ((1 << precisions) - 1)

def flatten(m):
    head, ([tail_limit,], tail_data) = m
    return jnp.concatenate([(head >> 3 * tail_prec).astype(tail_dtype),
                            (head >> 2 * tail_prec).astype(tail_dtype),
                            (head >> tail_prec).astype(tail_dtype),
                            head.astype(tail_dtype), tail_data[tail_limit:]])

def unflatten(arr, size, tail_capacity):
    head_highest, head_high, head_low, head_lowest, tail = jnp.split(
        arr, [size, 2 * size, 3 * size, 4 * size])
    head = (head_highest.astype(head_dtype) << 3 * tail_prec
            | head_high.astype(head_dtype)  << 2 * tail_prec
            | head_low.astype(head_dtype)   << 1 * tail_prec
            | head_lowest.astype(head_dtype))
    tail_limit = tail_capacity - tail.size
    tail = (jnp.array([tail_limit]),
            jnp.concatenate([jnp.zeros(tail_limit, tail_dtype), tail]))
    return head, tail

def message_equal(message1, message2):
    return jnp.all(flatten(message1) == flatten(message2))


##################### High level craystack-style API ##########################
Codec = namedtuple('Codec', ['push', 'pop'])

def NonUniform(enc_fun, dec_fun, precisions):
    precisions = atleast_1d(precisions)
    def push_(m, x):
        starts, freqs = map(atleast_1d, enc_fun(x))
        return push(m, starts, freqs, precisions)

    def pop_(m):
        cfs = peek(m, precisions)
        x = dec_fun(cfs)
        starts, freqs = map(atleast_1d, enc_fun(x))
        res = pop(m, cfs, starts, freqs, precisions), x
        return res
    return Codec(push_, pop_)

_uniform_enc_statfun = lambda s: (s, 1)
_uniform_dec_statfun = lambda cf: cf
def Uniform(precision):
    return NonUniform(_uniform_enc_statfun, _uniform_dec_statfun, precision)

def view_update(data, view_fun):
    item, view_transpose = view_fun(data), linear_transpose(view_fun, data)
    def update(new_item):
        diff, = view_transpose(tree_multimap(jnp.subtract, new_item, item))
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

def categorical_push(weights, prec, message, x):
    return substack(CategoricalUnsafe(weights, prec),
                    lambda x: x[0]).push(message, x)

def categorical_pop(weights, prec, message):
    return substack(CategoricalUnsafe(weights, prec),
                    lambda x: x[0]).pop(message)

def std_gaussian_bins(precision):
    return norm.ppf(jnp.linspace(0, 1, (1 << precision) + 1))

def std_gaussian_centres(precision):
    return norm.ppf((jnp.arange(1 << precision) + 0.5) / (1 << precision))

def _gaussian_cdf(mean, stdd, prior_prec, post_prec):
    def cdf(idx):
        x = std_gaussian_bins(prior_prec)[idx]
        return _nearest_uint(norm.cdf(x, mean, stdd) * (1 << post_prec))
    return cdf

def _gaussian_ppf(mean, stdd, prior_prec, post_prec):
    cdf = _gaussian_cdf(mean, stdd, prior_prec, post_prec)
    def ppf(cf):
        x = norm.ppf((cf + 0.5) / (1 << post_prec), mean, stdd)
        # Binary search is faster than using the actual gaussian cdf for the
        # precisions we typically use, however the cdf is O(1) whereas search
        # is O(precision), so for high precision cdf will be faster.
        idxs = jnp.digitize(x, std_gaussian_bins(prior_prec)) - 1
        # This loop works around an issue which is extremely rare when we use
        # float64 everywhere but is common if we work with float32: due to the
        # finite precision of floating point arithmetic, norm.[cdf,ppf] are not
        # perfectly inverse to each other.
        idxs_ = lax.while_loop(
            lambda idxs: ~jnp.all((cdf(idxs) <= cf) & (cf < cdf(idxs + 1))),
            lambda idxs: jnp.select(
                [cf < cdf(idxs), cf >= cdf(idxs + 1)],
                [idxs - 1,       idxs + 1           ], idxs),
            idxs)
        return idxs_
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


########################### Example usage #####################################
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    shape = 3  # The shape of data to be encoded

    # The compressed message is built up in a JAX DeviceArray. The
    # tail_capacity is the size of that array and will be an upper bound on
    # message size (in bytes). The message container is *unsafe* so exceeding
    # the capacity will result in silent failure (compressed data will be
    # over-written). You can use the stack_check function, as demonstrated
    # below, to check for overflow.
    tail_capacity = 100

    precision = 3
    n_data = 100
    data = jnp.array(rng.integers(0, 4, size=(n_data, shape)))

    # x ~ Categorical(1 / 8, 2 / 8, 3 / 8, 2 / 8)
    m = m_init = base_message(shape, tail_capacity)
    choose = partial(jnp.choose, mode='clip')

    # Using the high-level API requires an 'enc_fun' and 'dec_fun' to define a
    # model distribution. See
    # https://github.com/j-towns/craystack/blob/c676fd/craystack/codecs.py#L18-L56
    # for explanation.
    def enc_fun(x):
        return (choose(x, jnp.array([0, 1, 3, 6])),
                choose(x, jnp.array([1, 2, 3, 2])))

    def dec_fun(cf):
        return choose(cf, jnp.array([0, 1, 1, 2, 2, 2, 3, 3]))

    codec_push, codec_pop = NonUniform(enc_fun, dec_fun, precision)

    _, freqs = enc_fun(data)
    print("Exact entropy: " + str(np.sum(np.log2(8 / freqs))) + " bits.")

    # Encode
    message = base_message(shape, tail_capacity)
    def push_body(i, message):
        message = codec_push(message, data[n_data - i - 1])
        return message
    message = lax.fori_loop(0, n_data, push_body, message)

    _, stack = message
    stack_check(stack)  # Make sure overflow has not occurred

    # The flatten function converts the data structure we use to represent the
    # compressed message to a single DeviceArray of bytes.
    coded_arr = flatten(message)
    assert coded_arr.dtype == np.uint8
    print("Actual output length: " + str(8 * len(coded_arr)) + " bits.")

    # Decode
    def pop_body(i, carry):
        # We build up the decompressed data in the xs DeviceArray
        message, xs = carry
        message, x = codec_pop(message)
        return message, lax.dynamic_update_index_in_dim(xs, x, i, 0)
    message = unflatten(coded_arr, shape, tail_capacity)
    message, data_decoded = lax.fori_loop(
        0, n_data, pop_body, (message, jnp.zeros((n_data, shape), 'int32')))
    assert message_equal(m, m_init)
    np.testing.assert_array_equal(data, data_decoded)
