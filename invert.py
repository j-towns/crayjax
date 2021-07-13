from functools import partial
from dataclasses import make_dataclass

from jax.interpreters import xla
from jax.interpreters import partial_eval as pe
from jax.core import Primitive, ClosedJaxpr, eval_jaxpr, Jaxpr, JaxprEqn, Var
from jax import lax
from jax.util import safe_map, toposort
from jax.api_util import shaped_abstractify, flatten_fun_nokwargs, tree_flatten
import jax.linear_util as lu

map = safe_map


def custom_primitive(name, lowering, multiple_results=False):
    p = Primitive(name)
    p.multiple_results = multiple_results
    p.def_impl(partial(xla.apply_primitive, p))
    def p_abstract_eval(*avals, **params):
        if multiple_results:
            return pe.abstract_eval_fun(lowering, *avals, **params)
        else:
            out_aval, = pe.abstract_eval_fun(
                lambda *args, **params: [lowering(*args, **params)],
                *avals, **params)
            return out_aval
    p.def_abstract_eval(p_abstract_eval)
    xla.translations_with_avals[p] = xla.lower_fun(
        lowering, multiple_results=multiple_results, with_avals=True)
    return p

inverse_rules = {}

def invert_eqns(eqns, invars, outvars):
    active = set(outvars)
    new_eqns = []
    for e in reversed(eqns):
        if any(o in active for o in e.outvars):
            assert all(o in active for o in e.outvars)
            active = active.difference(e.outvars)
            new_eqn = inverse_rules[e.primitive](e)
            new_eqns.append(new_eqn)
            active = active.union(new_eqn.outvars)
        else:
            new_eqns.append(e)
    assert active == set(invars)
    return new_eqns

# Topologically sort the equations so that they're in a valid execution order.
def toposort_eqns(eqns, outvars):
    nodes = map(partial(make_dataclass('Node', ['parents', 'eqn']), None), eqns)
    outvar_to_node = {o: n for n, e in zip(nodes, eqns) for o in e.outvars}
    for n, e in zip(nodes, eqns):
        n.parents = [outvar_to_node[i] for i in e.invars if isinstance(i, Var)
                     and i in outvar_to_node]
    end_nodes = [outvar_to_node[o] for o in outvars if isinstance(o, Var) and o
                 in outvar_to_node]
    return [n.eqn for n in toposort(end_nodes)]

def invert_jaxpr(jaxpr: ClosedJaxpr):
    jaxpr, consts = jaxpr.jaxpr, jaxpr.consts
    invars, outvars, eqns = jaxpr.invars, jaxpr.outvars, jaxpr.eqns
    new_jaxpr = Jaxpr(
        jaxpr.constvars,
        jaxpr.outvars,
        jaxpr.invars,
        toposort_eqns(
            invert_eqns(jaxpr.eqns, jaxpr.invars, jaxpr.outvars),
            jaxpr.invars))
    return ClosedJaxpr(new_jaxpr, consts)

def invert(fun, dummy_input):
    inputs_flat, in_tree = tree_flatten([dummy_input])
    wrapped_fun, out_tree = flatten_fun_nokwargs(lu.wrap_init(fun), in_tree)
    # TODO: Add debug_info
    jaxpr, out_avals, consts = pe.trace_to_jaxpr_dynamic(
        wrapped_fun, map(shaped_abstractify, inputs_flat))
    ijaxpr = invert_jaxpr(ClosedJaxpr(jaxpr, consts))
    def inverse(output):
        out_flat, out_tree_ = tree_flatten(output)
        assert out_tree() == out_tree_
        assert map(shaped_abstractify, out_flat) == out_avals
        inputs = in_tree.unflatten(eval_jaxpr(ijaxpr.jaxpr, consts, *out_flat))
        assert len(inputs) == 1
        return inputs[0]
    return inverse

# Custom INVERTIBLE primitives
ineg_p = custom_primitive('ineg', lax.neg)

def ineg_p_irule(eqn):
    [i], [o], _, _, source_info = eqn
    return JaxprEqn([o], [i], lax.neg_p, {}, source_info)
inverse_rules[ineg_p] = ineg_p_irule

# Invertible API
def ineg(x):
    return ineg_p.bind(x)
