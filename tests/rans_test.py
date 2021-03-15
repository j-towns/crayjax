from functools import partial

from crayjax import rans
from numpy.testing import assert_equal
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

rng = np.random.default_rng(0)


def test_rans():
    shape = (3,)
    tail_capacity = 100
    precision = 3
    n_data = 100
    data = rng.integers(0, 4, size=(n_data, *shape))

    # x ~ Categorical(1 / 8, 2 / 8, 3 / 8, 2 / 8)
    m = m_init = rans.base_message(shape, tail_capacity)
    enc_fun = (lambda x: (np.choose(x, [0, 1, 3, 6]),
                          np.choose(x, [1, 2, 3, 2])))
    def dec_fun(cf):
        return np.choose(cf, [0, 1, 1, 2, 2, 2, 3, 3])
    codec_push, codec_pop = rans.rans((enc_fun, dec_fun, precision))
    _, freqs = enc_fun(data)
    print("Exact entropy: " + str(np.sum(np.log2(8 / freqs))) + " bits.")
    # Encode
    for x in reversed(data):
        m = codec_push(m, x)
    coded_arr = rans.flatten(m)
    assert coded_arr.dtype == np.uint16
    print("Actual output shape: " + str(16 * len(coded_arr)) + " bits.")

    # Decode
    m = rans.unflatten(coded_arr, shape, tail_capacity)
    data_decoded = []
    for _ in range(n_data):
        m, x = codec_pop(m)
        data_decoded.append(x)
    assert rans.message_equal(m, m_init)
    assert_equal(data, data_decoded)

def test_rans_jit():
    shape = (3,)
    tail_capacity = 100
    precision = 3
    n_data = 100
    data = rng.integers(0, 4, size=(n_data, *shape))

    # x ~ Categorical(1 / 8, 2 / 8, 3 / 8, 2 / 8)
    m = m_init = rans.base_message(shape, tail_capacity)
    choose = partial(jnp.choose, mode='clip')
    def enc_fun(x):
        assert is_tracing
        return (choose(x, jnp.array([0, 1, 3, 6])),
                choose(x, jnp.array([1, 2, 3, 2])))

    def dec_fun(cf):
        assert is_tracing
        return choose(cf, jnp.array([0, 1, 1, 2, 2, 2, 3, 3]))

    codec_push, codec_pop = rans.rans((enc_fun, dec_fun, precision))
    codec_push, codec_pop = map(jax.jit, (codec_push, codec_pop))

    is_tracing = True
    _, freqs = enc_fun(data)
    print("Exact entropy: " + str(np.sum(np.log2(8 / freqs))) + " bits.")

    # Encode
    m_ = codec_push(m, data[0])
    is_tracing = False
    for x in reversed(data):
        m = codec_push(m, x)
    coded_arr = rans.flatten(m)
    assert coded_arr.dtype == np.uint16
    print("Actual output shape: " + str(16 * len(coded_arr)) + " bits.")

    # Decode
    m = rans.unflatten(coded_arr, shape, tail_capacity)
    is_tracing = True
    codec_pop(m)
    is_tracing = False
    data_decoded = []
    for _ in range(n_data):
        m, x = codec_pop(m)
        data_decoded.append(x)
    assert rans.message_equal(m, m_init)
    assert_equal(data, data_decoded)

def test_rans_lax_fori_loop():
    shape = (3,)
    tail_capacity = 100
    precision = 3
    n_data = 100
    data = jnp.array(rng.integers(0, 4, size=(n_data, *shape)))

    # x ~ Categorical(1 / 8, 2 / 8, 3 / 8, 2 / 8)
    m = m_init = rans.base_message(shape, tail_capacity)
    choose = partial(jnp.choose, mode='clip')
    def enc_fun(x):
        return (choose(x, jnp.array([0, 1, 3, 6])),
                choose(x, jnp.array([1, 2, 3, 2])))

    def dec_fun(cf):
        return choose(cf, jnp.array([0, 1, 1, 2, 2, 2, 3, 3]))

    codec_push, codec_pop = rans.rans((enc_fun, dec_fun, precision))
    codec_push, codec_pop = map(jax.jit, (codec_push, codec_pop))

    _, freqs = enc_fun(data)
    print("Exact entropy: " + str(np.sum(np.log2(8 / freqs))) + " bits.")

    # Encode
    def push_body(i, carry):
        m = carry
        m = codec_push(m, data[n_data - i - 1])
        return m
    m = lax.fori_loop(0, n_data, push_body, m)
    coded_arr = rans.flatten(m)
    assert coded_arr.dtype == np.uint16
    print("Actual output shape: " + str(16 * len(coded_arr)) + " bits.")

    # Decode
    def pop_body(i, carry):
        m, xs = carry
        m, x = codec_pop(m)
        return m, lax.dynamic_update_index_in_dim(xs, x, i, 0)
    m = rans.unflatten(coded_arr, shape, tail_capacity)
    m, data_decoded = lax.fori_loop(0, n_data, pop_body,
                                    (m, jnp.zeros((n_data, *shape), 'int32')))
    assert rans.message_equal(m, m_init)
