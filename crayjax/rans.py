"""
Vectorized implementation of rANS based on https://arxiv.org/abs/1402.3392
"""
import numpy as np
import jax.numpy as jnp
from jax import lax


rng = np.random.default_rng(0)
head_prec, head_dtype = 32, 'uint32'
tail_prec, tail_dtype = 16, 'uint16'
head_min = 1 << head_prec - tail_prec
atleast_1d = lambda x: jnp.atleast_1d(x).astype(head_dtype)

def base_message(shape, tail_capacity):
    """
    Returns a base ANS message of given shape.
    """
    assert shape and np.prod(shape), 'Shape must be an int > 0' \
                                     'or tuple with length > 0.'
    head = jnp.full(shape, head_min, head_dtype)
    return head, empty_stack(tail_capacity)

def empty_stack(capacity):
    return jnp.array([capacity]), jnp.zeros(capacity, tail_dtype)

def stack_push(stack, idxs, arr):
    limit, data = stack
    return limit - idxs.sum(), lax.dynamic_update_slice(
        data, lax.sort_key_val(idxs, arr)[1], limit - arr.size)

def stack_pop(stack, idxs):
    limit, data = stack
    unsorted = lax.sort_key_val(idxs, jnp.arange(idxs.size))[1]
    limit = limit + idxs.sum()
    return (limit, data), lax.sort_key_val(
        unsorted, lax.dynamic_slice(data, limit - idxs.size, idxs.shape))[1]

def stack_check(stack):
    limit, _ = stack
    assert limit >= 0

def rans(model):
    enc_fun, dec_fun, precisions = model
    precisions = atleast_1d(precisions)
    def push(m, x):
        starts, freqs = enc_fun(x)
        starts, freqs = map(atleast_1d, (starts, freqs))
        head, tail = m
        idxs = head >= freqs << head_prec - precisions
        tail = stack_push(tail, idxs, head.astype(tail_dtype))
        head = jnp.where(idxs, head >> tail_prec, head)
        head_div_freqs, head_mod_freqs = jnp.divmod(head, freqs)
        return (head_div_freqs << precisions) + head_mod_freqs + starts, tail

    def pop(m):
        head_, tail_ = m
        cfs = head_ & ((1 << precisions) - 1)
        x = dec_fun(cfs)
        starts, freqs = enc_fun(x)
        starts, freqs = map(atleast_1d, (starts, freqs))
        head = freqs * (head_ >> precisions) + cfs - starts
        idxs = head < head_min
        tail, new_head = stack_pop(tail_, idxs)
        return (jnp.where(idxs, head << tail_prec | new_head, head), tail), x
    return push, pop

def flatten(m):
    head, ([tail_limit,], tail_data) = m
    return jnp.concatenate([(head >> tail_prec).astype(tail_dtype),
                            head.astype(tail_dtype), tail_data[tail_limit:]])

def unflatten(arr, head_shape, tail_capacity):
    size = np.prod(head_shape)
    head_high, head_low, tail = jnp.split(arr, [size, 2 * size])
    head = (head_high.astype(head_dtype) << tail_prec
            | head_low.astype(head_dtype))
    tail_limit = tail_capacity - tail.size
    tail = (jnp.array([tail_limit]),
            jnp.concatenate([jnp.zeros(tail_limit, tail_dtype), tail]))
    return head, tail

def message_equal(message1, message2):
    return jnp.all(flatten(message1) == flatten(message2))
