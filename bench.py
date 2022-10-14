from functools import partial

from time import time

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax, random, jit

import rans as rans
import craystack as cs


RNG_SEED = 0
TAIL_CAPACITY = 10_000_000  # 100 MB.
DATA_COUNT = 10_000_000
HEAD_SIZE = 4
NUM_TRIALS = 16

def time_thunk(thunk):
    t0 = time()
    thunk()
    return time() - t0

# Crayjax benchmarking
@partial(jit, static_argnums=(0, 1))
def compress_jax(head_size, tail_capacity, data):
    push, _ = rans.Uniform(8)
    message = rans.base_message(head_size, tail_capacity)
    message, _ = lax.scan(
        lambda message, datum: (push(message, datum), ()),
        message,
        data,
    )
    return message

@partial(jit, static_argnums=0)
def decompress_jax(data_count, message):
    _, pop = rans.Uniform(8)
    _, data = lax.scan(
        lambda message, _: pop(message),
        message,
        (),
        data_count,
    )
    return jnp.flip(data, 0)

def bench_jax(tail_capacity, data_count, rng_seed, head_size, num_trials):
    data = jnp.zeros((data_count, head_size), 'uint8')
    data_size = data.size / 1_000_000  # Size in MB

    # First run to compile
    compressed = rans.flatten(compress_jax(head_size, tail_capacity, data))

    # Then time

    compress_speeds = [data_size / time_thunk(
        lambda: rans.flatten(
            compress_jax(head_size, tail_capacity, data)).block_until_ready())
              for _ in range(num_trials)]

    decompressed = decompress_jax(data_count, rans.unflatten(
        compressed, head_size, tail_capacity))

    assert jnp.all(data == decompressed)

    decompress_speeds = [data_size / time_thunk(lambda: decompress_jax(
        data_count, rans.unflatten(
            compressed, head_size, tail_capacity)).block_until_ready())
                         for _ in range(num_trials)]

    return data_size, np.mean(compress_speeds), np.mean(decompress_speeds)

# Craystack benchmarking
def compress_cs(head_size, data):
    push, _ = cs.Uniform(8)
    message = cs.base_message((head_size,))
    for datum in data:
        message, = push(message, datum)
    return cs.flatten(message)

def decompress_cs(data_count, head_size, message_flat):
    message = cs.unflatten(message_flat, (head_size,))
    _, pop = cs.Uniform(8)
    data = []
    for _ in range(data_count):
        message, datum = pop(message)
        data.append(datum)
    return np.flip(np.array(data), 0)

def bench_cs(tail_capacity, data_count, rng_seed, head_size, num_trials):
    rng = np.random.default_rng(rng_seed)
    data = rng.integers(0, 1 << 8, (data_count, head_size), dtype="int32")
    data_size = data.size / 1_000_000  # Size in MB

    # First run to compile
    compressed = compress_cs(head_size, data)

    # Then time
    compress_speeds = [data_size / time_thunk(
        lambda: compress_cs(head_size, data))
              for _ in range(num_trials)]

    decompressed = decompress_cs(data_count, head_size, compressed)

    assert jnp.all(data == decompressed)

    decompress_speeds = [data_size / time_thunk(lambda: decompress_cs(
        data_count, head_size, compressed))
                         for _ in range(num_trials)]

    return data_size, np.mean(compress_speeds), np.mean(decompress_speeds)

if __name__ == "__main__":
    print(bench_jax(TAIL_CAPACITY, DATA_COUNT, RNG_SEED, HEAD_SIZE, NUM_TRIALS))
    # print(bench_cs(TAIL_CAPACITY, DATA_COUNT, RNG_SEED, HEAD_SIZE, NUM_TRIALS))
