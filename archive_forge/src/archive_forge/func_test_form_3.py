from sympy.core.backend import symbols, Matrix, atan, zeros
from sympy.simplify.simplify import simplify
from sympy.physics.mechanics import (dynamicsymbols, Particle, Point,
from sympy.testing.pytest import raises
def test_form_3():
    symsystem3 = SymbolicSystem(states, dyn_implicit_rhs, mass_matrix=dyn_implicit_mat, coordinate_derivatives=kin_explicit_rhs, alg_con=alg_con, coord_idxs=coord_idxs, speed_idxs=speed_idxs, bodies=bodies, loads=loads)
    assert symsystem3.coordinates == Matrix([x, y])
    assert symsystem3.speeds == Matrix([u, v])
    assert symsystem3.states == Matrix([x, y, u, v, lam])
    assert symsystem3.alg_con == [4]
    inter1 = kin_explicit_rhs
    inter2 = dyn_implicit_rhs
    assert simplify(symsystem3.kin_explicit_rhs - inter1) == zeros(2, 1)
    assert simplify(symsystem3.dyn_implicit_mat - dyn_implicit_mat) == zeros(3)
    assert simplify(symsystem3.dyn_implicit_rhs - inter2) == zeros(3, 1)
    inter = comb_implicit_rhs
    assert simplify(symsystem3.comb_implicit_rhs - inter) == zeros(5, 1)
    assert simplify(symsystem3.comb_implicit_mat - comb_implicit_mat) == zeros(5)
    inter = comb_explicit_rhs
    symsystem3.compute_explicit_form()
    assert simplify(symsystem3.comb_explicit_rhs - inter) == zeros(5, 1)
    assert set(symsystem3.dynamic_symbols()) == {y, v, lam, u, x}
    assert type(symsystem3.dynamic_symbols()) == tuple
    assert set(symsystem3.constant_symbols()) == {l, g, m}
    assert type(symsystem3.constant_symbols()) == tuple
    assert symsystem3.output_eqns == {}
    assert symsystem3.bodies == (Pa,)
    assert symsystem3.loads == ((P, g * m * N.x),)