from functools import partial

import rans as rans
from numpy.testing import assert_equal
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax


rng = np.random.default_rng(1)
default_capacity = 1024

def check_codec_unsafe(head_size, codec, capacity=default_capacity):
    message = rans.base_message(head_size, capacity)
    message = rans.Uniform(16).push(
        message, rng.integers(1 << 16, size=head_size))
    message = rans.Uniform(16).push(
        message, rng.integers(1 << 16, size=head_size))
    push, pop = codec
    message_, data = pop(message)
    message__ = push(message_, data)
    assert rans.message_equal(message, message__)
    message___, data_ = pop(message__)
    assert np.all(data == data_)
    assert rans.message_equal(message_, message___)

def check_codec(head_size, codec, data, capacity=default_capacity):
    assert data.shape == (head_size,)
    message = rans.base_message(head_size, capacity)
    push, pop = codec
    message_, data_ = pop(push(message, data))
    assert rans.message_equal(rans.base_message(head_size, capacity), message_)
    assert np.all(data == data_)

def test_rans_simple():
    size = 3
    tail_capacity = 100
    precision = 24
    n_data = 10
    data = rng.integers(0, 4, size=(n_data, size))

    # x ~ Categorical(1 / 8, 2 / 8, 3 / 8, 2 / 8)
    m = m_init = rans.base_message(size, tail_capacity)
    enc_fun = (lambda x: (jnp.choose(x, [0, 1, 3, 6]),
                          jnp.choose(x, [1, 2, 3, (1 << 24) - 6])))
    def dec_fun(cf):
        return jnp.where(
            cf < 6,
            jnp.choose(cf, [0, 1, 1, 2, 2, 2], mode='clip'),
            3)
    codec_push, codec_pop = rans.NonUniform(enc_fun, dec_fun, precision)
    _, freqs = enc_fun(data)
    # Encode
    for x in reversed(data):
        m = codec_push(m, x)
    coded_arr = rans.flatten(m)
    assert coded_arr.dtype == np.uint8

    # Decode
    m = rans.unflatten(coded_arr, size, tail_capacity)
    data_decoded = []
    for _ in range(n_data):
        m, x = codec_pop(m)
        data_decoded.append(x)
    assert rans.message_equal(m, m_init)
    assert_equal(data, data_decoded)

def test_rans_jit():
    size = 3
    tail_capacity = 100
    precision = 3
    n_data = 100
    data = rng.integers(0, 4, size=(n_data, size))

    # x ~ Categorical(1 / 8, 2 / 8, 3 / 8, 2 / 8)
    m = m_init = rans.base_message(size, tail_capacity)
    choose = partial(jnp.choose, mode='clip')
    def enc_fun(x):
        assert is_tracing
        return (choose(x, jnp.array([0, 1, 3, 6])),
                choose(x, jnp.array([1, 2, 3, 2])))

    def dec_fun(cf):
        assert is_tracing
        return choose(cf, jnp.array([0, 1, 1, 2, 2, 2, 3, 3]))

    codec_push, codec_pop = rans.NonUniform(enc_fun, dec_fun, precision)
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
    assert coded_arr.dtype == np.uint8
    print("Actual output shape: " + str(16 * len(coded_arr)) + " bits.")

    # Decode
    m = rans.unflatten(coded_arr, size, tail_capacity)
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
    size = 3
    tail_capacity = 100
    precision = 3
    n_data = 100
    data = jnp.array(rng.integers(0, 4, size=(n_data, size)))

    # x ~ Categorical(1 / 8, 2 / 8, 3 / 8, 2 / 8)
    m = m_init = rans.base_message(size, tail_capacity)
    choose = partial(jnp.choose, mode='clip')
    def enc_fun(x):
        return (choose(x, jnp.array([0, 1, 3, 6])),
                choose(x, jnp.array([1, 2, 3, 2])))

    def dec_fun(cf):
        return choose(cf, jnp.array([0, 1, 1, 2, 2, 2, 3, 3]))

    codec_push, codec_pop = rans.NonUniform(enc_fun, dec_fun, precision)

    _, freqs = enc_fun(data)

    # Encode
    def push_body(i, carry):
        m = carry
        m = codec_push(m, data[n_data - i - 1])
        return m
    m = lax.fori_loop(0, n_data, push_body, m)
    coded_arr = rans.flatten(m)
    assert coded_arr.dtype == np.uint8

    # Decode
    def pop_body(i, carry):
        m, xs = carry
        m, x = codec_pop(m)
        return m, lax.dynamic_update_index_in_dim(xs, x, i, 0)
    m = rans.unflatten(coded_arr, size, tail_capacity)
    m, data_decoded = lax.fori_loop(0, n_data, pop_body,
                                    (m, jnp.zeros((n_data, size), 'int32')))
    assert rans.message_equal(m, m_init)

def test_uniform():
    precision = 4
    size = 3
    data = rng.integers(precision, size=size, dtype="uint64")
    check_codec(size, rans.Uniform(precision), data)

def test_substack():
    prec = 4
    message = rans.base_message(16, 50)
    data = rng.integers(1 << prec, size=8, dtype='uint64')
    view_split = lambda h: jnp.split(h, 2)
    view_left  = lambda h: view_split(h)[0]
    view_right = lambda h: view_split(h)[1]
    append, pop = rans.substack(rans.Uniform(prec), view_left)
    message_ = append(message, data)
    np.testing.assert_array_equal(view_right(message_[0]),
                                  view_right(message[0]))
    message_, data_ = pop(message_)
    assert rans.message_equal(rans.base_message(16, 50), message_)
    np.testing.assert_equal(data, data_)

    append, pop = jax.jit(append), jax.jit(pop)
    message = rans.base_message(16, 50)
    message_ = append(message, data)
    np.testing.assert_array_equal(view_right(message_[0]),
                                  view_right(message[0]))
    message_, data_ = pop(message_)
    assert rans.message_equal(message, message_)
    np.testing.assert_equal(data, data_)

    message = rans.base_message(16, 50)
    message_ = append(message, data)
    np.testing.assert_array_equal(view_right(message_[0]),
                                  view_right(message[0]))
    message_, data_ = pop(message_)
    assert rans.message_equal(message, message_)
    np.testing.assert_equal(data, data_)

def test_categorical_unsafe():
    precision = 4
    size = 5
    weights = rng.random((size, 4))
    check_codec_unsafe(size, rans.CategoricalUnsafe(weights, precision))

def test_bernoulli():
    precision = 2
    size = 100
    p = rng.random(size)
    data = np.uint64(rng.random(size) < p)
    check_codec(size, rans.Bernoulli(p, precision), data)

def test_gaussian_stdbins():
    bin_precision = 8
    coding_precision = 12
    batch_size = 1000

    means = jnp.array(rng.normal(size=batch_size))
    stdds = jnp.array(np.exp(rng.normal(size=batch_size)))

    check_codec_unsafe(batch_size, rans.DiagGaussian_StdBins(
            means, stdds, coding_precision, bin_precision))
