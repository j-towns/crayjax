from functools import partial

from jax import lax
from jax import jit
from invert import custom_primitive
import invert


def test_custom_primitive():
    custom_add_p = custom_primitive('custom_add', lambda a, b: lax.add(a, b),
                                    multiple_results=False)
    assert custom_add_p.bind(1, 2) == 3
    assert jit(custom_add_p.bind)(1, 2) == 3

def test_custom_primitive_multiple_results():
    custom_add_p = custom_primitive(
        'custom_add_sub',
        lambda a, b: [lax.add(a, b), lax.sub(a, b)],
        multiple_results=True)
    assert custom_add_p.bind(1, 2) == [3, -1]
    assert jit(custom_add_p.bind)(1, 2) == [3, -1]

def test_custom_primitive_containing_jit():
    custom_add_p = custom_primitive('custom_add',
                                    lambda a, b: jit(lax.add)(a, b),
                                    multiple_results=False)
    assert custom_add_p.bind(1, 2) == 3
    assert jit(custom_add_p.bind)(1, 2) == 3

def test_custom_primitive_with_params():
    custom_reshape_p = custom_primitive('custom_reshape',
                                        lambda a, shape: lax.reshape(a, shape),
                                        multiple_results=False)
    assert custom_reshape_p.bind(1, shape=(1,)).shape == (1,)
    assert jit(partial(custom_reshape_p.bind, shape=(1,)))(1).shape == (1,)

def test_invert():
    def f(x):
        return invert.ineg(x)

    assert f(1) == -1
    assert invert.invert(f, 1)(1) == -1
