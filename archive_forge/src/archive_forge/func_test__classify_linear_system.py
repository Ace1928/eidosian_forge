from sympy.core.function import (Derivative, Function, diff)
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.hyperbolic import sinh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import Matrix
from sympy.core.containers import Tuple
from sympy.functions import exp, cos, sin, log, Ci, Si, erf, erfi
from sympy.matrices import dotprodsimp, NonSquareMatrixError
from sympy.solvers.ode import dsolve
from sympy.solvers.ode.ode import constant_renumber
from sympy.solvers.ode.subscheck import checksysodesol
from sympy.solvers.ode.systems import (_classify_linear_system, linear_ode_to_matrix,
from sympy.functions import airyai, airybi
from sympy.integrals.integrals import Integral
from sympy.simplify.ratsimp import ratsimp
from sympy.testing.pytest import ON_CI, raises, slow, skip, XFAIL
def test__classify_linear_system():
    x, y, z, w = symbols('x, y, z, w', cls=Function)
    t, k, l = symbols('t k l')
    x1 = diff(x(t), t)
    y1 = diff(y(t), t)
    z1 = diff(z(t), t)
    w1 = diff(w(t), t)
    x2 = diff(x(t), t, t)
    y2 = diff(y(t), t, t)
    funcs = [x(t), y(t)]
    funcs_2 = funcs + [z(t), w(t)]
    eqs_1 = (5 * x1 + 12 * x(t) - 6 * y(t), 2 * y1 - 11 * t * x(t) + 3 * y(t) + t)
    assert _classify_linear_system(eqs_1, funcs, t) is None
    eqs_2 = (5 * x1 ** 2 + 12 * x(t) - 6 * y(t), 2 * y1 - 11 * t * x(t) + 3 * y(t) + t)
    sol2 = {'is_implicit': True, 'canon_eqs': [[Eq(Derivative(x(t), t), -sqrt(-12 * x(t) / 5 + 6 * y(t) / 5)), Eq(Derivative(y(t), t), 11 * t * x(t) / 2 - t / 2 - 3 * y(t) / 2)], [Eq(Derivative(x(t), t), sqrt(-12 * x(t) / 5 + 6 * y(t) / 5)), Eq(Derivative(y(t), t), 11 * t * x(t) / 2 - t / 2 - 3 * y(t) / 2)]]}
    assert _classify_linear_system(eqs_2, funcs, t) == sol2
    eqs_2_1 = [Eq(Derivative(x(t), t), -sqrt(-12 * x(t) / 5 + 6 * y(t) / 5)), Eq(Derivative(y(t), t), 11 * t * x(t) / 2 - t / 2 - 3 * y(t) / 2)]
    assert _classify_linear_system(eqs_2_1, funcs, t) is None
    eqs_2_2 = [Eq(Derivative(x(t), t), sqrt(-12 * x(t) / 5 + 6 * y(t) / 5)), Eq(Derivative(y(t), t), 11 * t * x(t) / 2 - t / 2 - 3 * y(t) / 2)]
    assert _classify_linear_system(eqs_2_2, funcs, t) is None
    eqs_3 = (5 * x1 + 12 * x(t) - 6 * y(t), 2 * y1 - 11 * x(t) + 3 * y(t), 5 * w1 + z(t), z1 + w(t))
    answer_3 = {'no_of_equation': 4, 'eq': (12 * x(t) - 6 * y(t) + 5 * Derivative(x(t), t), -11 * x(t) + 3 * y(t) + 2 * Derivative(y(t), t), z(t) + 5 * Derivative(w(t), t), w(t) + Derivative(z(t), t)), 'func': [x(t), y(t), z(t), w(t)], 'order': {x(t): 1, y(t): 1, z(t): 1, w(t): 1}, 'is_linear': True, 'is_constant': True, 'is_homogeneous': True, 'func_coeff': -Matrix([[Rational(12, 5), Rational(-6, 5), 0, 0], [Rational(-11, 2), Rational(3, 2), 0, 0], [0, 0, 0, 1], [0, 0, Rational(1, 5), 0]]), 'type_of_equation': 'type1', 'is_general': True}
    assert _classify_linear_system(eqs_3, funcs_2, t) == answer_3
    eqs_4 = (5 * x1 + 12 * x(t) - 6 * y(t), 2 * y1 - 11 * x(t) + 3 * y(t), z1 - w(t), w1 - z(t))
    answer_4 = {'no_of_equation': 4, 'eq': (12 * x(t) - 6 * y(t) + 5 * Derivative(x(t), t), -11 * x(t) + 3 * y(t) + 2 * Derivative(y(t), t), -w(t) + Derivative(z(t), t), -z(t) + Derivative(w(t), t)), 'func': [x(t), y(t), z(t), w(t)], 'order': {x(t): 1, y(t): 1, z(t): 1, w(t): 1}, 'is_linear': True, 'is_constant': True, 'is_homogeneous': True, 'func_coeff': -Matrix([[Rational(12, 5), Rational(-6, 5), 0, 0], [Rational(-11, 2), Rational(3, 2), 0, 0], [0, 0, 0, -1], [0, 0, -1, 0]]), 'type_of_equation': 'type1', 'is_general': True}
    assert _classify_linear_system(eqs_4, funcs_2, t) == answer_4
    eqs_5 = (5 * x1 + 12 * x(t) - 6 * y(t) + x2, 2 * y1 - 11 * x(t) + 3 * y(t), z1 - w(t), w1 - z(t))
    answer_5 = {'no_of_equation': 4, 'eq': (12 * x(t) - 6 * y(t) + 5 * Derivative(x(t), t) + Derivative(x(t), (t, 2)), -11 * x(t) + 3 * y(t) + 2 * Derivative(y(t), t), -w(t) + Derivative(z(t), t), -z(t) + Derivative(w(t), t)), 'func': [x(t), y(t), z(t), w(t)], 'order': {x(t): 2, y(t): 1, z(t): 1, w(t): 1}, 'is_linear': True, 'is_homogeneous': True, 'is_general': True, 'type_of_equation': 'type0', 'is_higher_order': True}
    assert _classify_linear_system(eqs_5, funcs_2, t) == answer_5
    eqs_6 = (Eq(x1, 3 * y(t) - 11 * z(t)), Eq(y1, 7 * z(t) - 3 * x(t)), Eq(z1, 11 * x(t) - 7 * y(t)))
    answer_6 = {'no_of_equation': 3, 'eq': (Eq(Derivative(x(t), t), 3 * y(t) - 11 * z(t)), Eq(Derivative(y(t), t), -3 * x(t) + 7 * z(t)), Eq(Derivative(z(t), t), 11 * x(t) - 7 * y(t))), 'func': [x(t), y(t), z(t)], 'order': {x(t): 1, y(t): 1, z(t): 1}, 'is_linear': True, 'is_constant': True, 'is_homogeneous': True, 'func_coeff': -Matrix([[0, -3, 11], [3, 0, -7], [-11, 7, 0]]), 'type_of_equation': 'type1', 'is_general': True}
    assert _classify_linear_system(eqs_6, funcs_2[:-1], t) == answer_6
    eqs_7 = (Eq(x1, y(t)), Eq(y1, x(t)))
    answer_7 = {'no_of_equation': 2, 'eq': (Eq(Derivative(x(t), t), y(t)), Eq(Derivative(y(t), t), x(t))), 'func': [x(t), y(t)], 'order': {x(t): 1, y(t): 1}, 'is_linear': True, 'is_constant': True, 'is_homogeneous': True, 'func_coeff': -Matrix([[0, -1], [-1, 0]]), 'type_of_equation': 'type1', 'is_general': True}
    assert _classify_linear_system(eqs_7, funcs, t) == answer_7
    eqs_8 = (Eq(x1, 21 * x(t)), Eq(y1, 17 * x(t) + 3 * y(t)), Eq(z1, 5 * x(t) + 7 * y(t) + 9 * z(t)))
    answer_8 = {'no_of_equation': 3, 'eq': (Eq(Derivative(x(t), t), 21 * x(t)), Eq(Derivative(y(t), t), 17 * x(t) + 3 * y(t)), Eq(Derivative(z(t), t), 5 * x(t) + 7 * y(t) + 9 * z(t))), 'func': [x(t), y(t), z(t)], 'order': {x(t): 1, y(t): 1, z(t): 1}, 'is_linear': True, 'is_constant': True, 'is_homogeneous': True, 'func_coeff': -Matrix([[-21, 0, 0], [-17, -3, 0], [-5, -7, -9]]), 'type_of_equation': 'type1', 'is_general': True}
    assert _classify_linear_system(eqs_8, funcs_2[:-1], t) == answer_8
    eqs_9 = (Eq(x1, 4 * x(t) + 5 * y(t) + 2 * z(t)), Eq(y1, x(t) + 13 * y(t) + 9 * z(t)), Eq(z1, 32 * x(t) + 41 * y(t) + 11 * z(t)))
    answer_9 = {'no_of_equation': 3, 'eq': (Eq(Derivative(x(t), t), 4 * x(t) + 5 * y(t) + 2 * z(t)), Eq(Derivative(y(t), t), x(t) + 13 * y(t) + 9 * z(t)), Eq(Derivative(z(t), t), 32 * x(t) + 41 * y(t) + 11 * z(t))), 'func': [x(t), y(t), z(t)], 'order': {x(t): 1, y(t): 1, z(t): 1}, 'is_linear': True, 'is_constant': True, 'is_homogeneous': True, 'func_coeff': -Matrix([[-4, -5, -2], [-1, -13, -9], [-32, -41, -11]]), 'type_of_equation': 'type1', 'is_general': True}
    assert _classify_linear_system(eqs_9, funcs_2[:-1], t) == answer_9
    eqs_10 = (Eq(3 * x1, 4 * 5 * (y(t) - z(t))), Eq(4 * y1, 3 * 5 * (z(t) - x(t))), Eq(5 * z1, 3 * 4 * (x(t) - y(t))))
    answer_10 = {'no_of_equation': 3, 'eq': (Eq(3 * Derivative(x(t), t), 20 * y(t) - 20 * z(t)), Eq(4 * Derivative(y(t), t), -15 * x(t) + 15 * z(t)), Eq(5 * Derivative(z(t), t), 12 * x(t) - 12 * y(t))), 'func': [x(t), y(t), z(t)], 'order': {x(t): 1, y(t): 1, z(t): 1}, 'is_linear': True, 'is_constant': True, 'is_homogeneous': True, 'func_coeff': -Matrix([[0, Rational(-20, 3), Rational(20, 3)], [Rational(15, 4), 0, Rational(-15, 4)], [Rational(-12, 5), Rational(12, 5), 0]]), 'type_of_equation': 'type1', 'is_general': True}
    assert _classify_linear_system(eqs_10, funcs_2[:-1], t) == answer_10
    eq11 = (Eq(x1, 3 * y(t) - 11 * z(t)), Eq(y1, 7 * z(t) - 3 * x(t)), Eq(z1, 11 * x(t) - 7 * y(t)))
    sol11 = {'no_of_equation': 3, 'eq': (Eq(Derivative(x(t), t), 3 * y(t) - 11 * z(t)), Eq(Derivative(y(t), t), -3 * x(t) + 7 * z(t)), Eq(Derivative(z(t), t), 11 * x(t) - 7 * y(t))), 'func': [x(t), y(t), z(t)], 'order': {x(t): 1, y(t): 1, z(t): 1}, 'is_linear': True, 'is_constant': True, 'is_homogeneous': True, 'func_coeff': -Matrix([[0, -3, 11], [3, 0, -7], [-11, 7, 0]]), 'type_of_equation': 'type1', 'is_general': True}
    assert _classify_linear_system(eq11, funcs_2[:-1], t) == sol11
    eq12 = (Eq(Derivative(x(t), t), y(t)), Eq(Derivative(y(t), t), x(t)))
    sol12 = {'no_of_equation': 2, 'eq': (Eq(Derivative(x(t), t), y(t)), Eq(Derivative(y(t), t), x(t))), 'func': [x(t), y(t)], 'order': {x(t): 1, y(t): 1}, 'is_linear': True, 'is_constant': True, 'is_homogeneous': True, 'func_coeff': -Matrix([[0, -1], [-1, 0]]), 'type_of_equation': 'type1', 'is_general': True}
    assert _classify_linear_system(eq12, [x(t), y(t)], t) == sol12
    eq13 = (Eq(Derivative(x(t), t), 21 * x(t)), Eq(Derivative(y(t), t), 17 * x(t) + 3 * y(t)), Eq(Derivative(z(t), t), 5 * x(t) + 7 * y(t) + 9 * z(t)))
    sol13 = {'no_of_equation': 3, 'eq': (Eq(Derivative(x(t), t), 21 * x(t)), Eq(Derivative(y(t), t), 17 * x(t) + 3 * y(t)), Eq(Derivative(z(t), t), 5 * x(t) + 7 * y(t) + 9 * z(t))), 'func': [x(t), y(t), z(t)], 'order': {x(t): 1, y(t): 1, z(t): 1}, 'is_linear': True, 'is_constant': True, 'is_homogeneous': True, 'func_coeff': -Matrix([[-21, 0, 0], [-17, -3, 0], [-5, -7, -9]]), 'type_of_equation': 'type1', 'is_general': True}
    assert _classify_linear_system(eq13, [x(t), y(t), z(t)], t) == sol13
    eq14 = (Eq(Derivative(x(t), t), 4 * x(t) + 5 * y(t) + 2 * z(t)), Eq(Derivative(y(t), t), x(t) + 13 * y(t) + 9 * z(t)), Eq(Derivative(z(t), t), 32 * x(t) + 41 * y(t) + 11 * z(t)))
    sol14 = {'no_of_equation': 3, 'eq': (Eq(Derivative(x(t), t), 4 * x(t) + 5 * y(t) + 2 * z(t)), Eq(Derivative(y(t), t), x(t) + 13 * y(t) + 9 * z(t)), Eq(Derivative(z(t), t), 32 * x(t) + 41 * y(t) + 11 * z(t))), 'func': [x(t), y(t), z(t)], 'order': {x(t): 1, y(t): 1, z(t): 1}, 'is_linear': True, 'is_constant': True, 'is_homogeneous': True, 'func_coeff': -Matrix([[-4, -5, -2], [-1, -13, -9], [-32, -41, -11]]), 'type_of_equation': 'type1', 'is_general': True}
    assert _classify_linear_system(eq14, [x(t), y(t), z(t)], t) == sol14
    eq15 = (Eq(3 * Derivative(x(t), t), 20 * y(t) - 20 * z(t)), Eq(4 * Derivative(y(t), t), -15 * x(t) + 15 * z(t)), Eq(5 * Derivative(z(t), t), 12 * x(t) - 12 * y(t)))
    sol15 = {'no_of_equation': 3, 'eq': (Eq(3 * Derivative(x(t), t), 20 * y(t) - 20 * z(t)), Eq(4 * Derivative(y(t), t), -15 * x(t) + 15 * z(t)), Eq(5 * Derivative(z(t), t), 12 * x(t) - 12 * y(t))), 'func': [x(t), y(t), z(t)], 'order': {x(t): 1, y(t): 1, z(t): 1}, 'is_linear': True, 'is_constant': True, 'is_homogeneous': True, 'func_coeff': -Matrix([[0, Rational(-20, 3), Rational(20, 3)], [Rational(15, 4), 0, Rational(-15, 4)], [Rational(-12, 5), Rational(12, 5), 0]]), 'type_of_equation': 'type1', 'is_general': True}
    assert _classify_linear_system(eq15, [x(t), y(t), z(t)], t) == sol15
    eq1 = (Eq(diff(x(t), t), x(t) + y(t) + 9), Eq(diff(y(t), t), 2 * x(t) + 5 * y(t) + 23))
    sol1 = {'no_of_equation': 2, 'eq': (Eq(Derivative(x(t), t), x(t) + y(t) + 9), Eq(Derivative(y(t), t), 2 * x(t) + 5 * y(t) + 23)), 'func': [x(t), y(t)], 'order': {x(t): 1, y(t): 1}, 'is_linear': True, 'is_constant': True, 'is_homogeneous': False, 'is_general': True, 'func_coeff': -Matrix([[-1, -1], [-2, -5]]), 'rhs': Matrix([[9], [23]]), 'type_of_equation': 'type2'}
    assert _classify_linear_system(eq1, funcs, t) == sol1
    eq1 = (Eq(diff(x(t), t), 5 * t * x(t) + 2 * y(t)), Eq(diff(y(t), t), 2 * x(t) + 5 * t * y(t)))
    sol1 = {'no_of_equation': 2, 'eq': (Eq(Derivative(x(t), t), 5 * t * x(t) + 2 * y(t)), Eq(Derivative(y(t), t), 5 * t * y(t) + 2 * x(t))), 'func': [x(t), y(t)], 'order': {x(t): 1, y(t): 1}, 'is_linear': True, 'is_constant': False, 'is_homogeneous': True, 'func_coeff': -Matrix([[-5 * t, -2], [-2, -5 * t]]), 'commutative_antiderivative': Matrix([[5 * t ** 2 / 2, 2 * t], [2 * t, 5 * t ** 2 / 2]]), 'type_of_equation': 'type3', 'is_general': True}
    assert _classify_linear_system(eq1, funcs, t) == sol1
    eq1 = [Eq(x1, x(t) + t * y(t) + t), Eq(y1, t * x(t) + y(t))]
    sol1 = {'no_of_equation': 2, 'eq': [Eq(Derivative(x(t), t), t * y(t) + t + x(t)), Eq(Derivative(y(t), t), t * x(t) + y(t))], 'func': [x(t), y(t)], 'order': {x(t): 1, y(t): 1}, 'is_linear': True, 'is_constant': False, 'is_homogeneous': False, 'is_general': True, 'func_coeff': -Matrix([[-1, -t], [-t, -1]]), 'commutative_antiderivative': Matrix([[t, t ** 2 / 2], [t ** 2 / 2, t]]), 'rhs': Matrix([[t], [0]]), 'type_of_equation': 'type4'}
    assert _classify_linear_system(eq1, funcs, t) == sol1
    eq2 = [Eq(x1, t * x(t) + t * y(t) + t), Eq(y1, t * x(t) + t * y(t) + cos(t))]
    sol2 = {'no_of_equation': 2, 'eq': [Eq(Derivative(x(t), t), t * x(t) + t * y(t) + t), Eq(Derivative(y(t), t), t * x(t) + t * y(t) + cos(t))], 'func': [x(t), y(t)], 'order': {x(t): 1, y(t): 1}, 'is_linear': True, 'is_homogeneous': False, 'is_general': True, 'rhs': Matrix([[t], [cos(t)]]), 'func_coeff': Matrix([[t, t], [t, t]]), 'is_constant': False, 'type_of_equation': 'type4', 'commutative_antiderivative': Matrix([[t ** 2 / 2, t ** 2 / 2], [t ** 2 / 2, t ** 2 / 2]])}
    assert _classify_linear_system(eq2, funcs, t) == sol2
    eq3 = [Eq(x1, t * (x(t) + y(t) + z(t) + 1)), Eq(y1, t * (x(t) + y(t) + z(t))), Eq(z1, t * (x(t) + y(t) + z(t)))]
    sol3 = {'no_of_equation': 3, 'eq': [Eq(Derivative(x(t), t), t * (x(t) + y(t) + z(t) + 1)), Eq(Derivative(y(t), t), t * (x(t) + y(t) + z(t))), Eq(Derivative(z(t), t), t * (x(t) + y(t) + z(t)))], 'func': [x(t), y(t), z(t)], 'order': {x(t): 1, y(t): 1, z(t): 1}, 'is_linear': True, 'is_constant': False, 'is_homogeneous': False, 'is_general': True, 'func_coeff': -Matrix([[-t, -t, -t], [-t, -t, -t], [-t, -t, -t]]), 'commutative_antiderivative': Matrix([[t ** 2 / 2, t ** 2 / 2, t ** 2 / 2], [t ** 2 / 2, t ** 2 / 2, t ** 2 / 2], [t ** 2 / 2, t ** 2 / 2, t ** 2 / 2]]), 'rhs': Matrix([[t], [0], [0]]), 'type_of_equation': 'type4'}
    assert _classify_linear_system(eq3, funcs_2[:-1], t) == sol3
    eq4 = [Eq(x1, x(t) + y(t) + t * z(t) + 1), Eq(y1, x(t) + t * y(t) + z(t) + 10), Eq(z1, t * x(t) + y(t) + z(t) + t)]
    sol4 = {'no_of_equation': 3, 'eq': [Eq(Derivative(x(t), t), t * z(t) + x(t) + y(t) + 1), Eq(Derivative(y(t), t), t * y(t) + x(t) + z(t) + 10), Eq(Derivative(z(t), t), t * x(t) + t + y(t) + z(t))], 'func': [x(t), y(t), z(t)], 'order': {x(t): 1, y(t): 1, z(t): 1}, 'is_linear': True, 'is_constant': False, 'is_homogeneous': False, 'is_general': True, 'func_coeff': -Matrix([[-1, -1, -t], [-1, -t, -1], [-t, -1, -1]]), 'commutative_antiderivative': Matrix([[t, t, t ** 2 / 2], [t, t ** 2 / 2, t], [t ** 2 / 2, t, t]]), 'rhs': Matrix([[1], [10], [t]]), 'type_of_equation': 'type4'}
    assert _classify_linear_system(eq4, funcs_2[:-1], t) == sol4
    sum_terms = t * (x(t) + y(t) + z(t) + w(t))
    eq5 = [Eq(x1, sum_terms), Eq(y1, sum_terms), Eq(z1, sum_terms + 1), Eq(w1, sum_terms)]
    sol5 = {'no_of_equation': 4, 'eq': [Eq(Derivative(x(t), t), t * (w(t) + x(t) + y(t) + z(t))), Eq(Derivative(y(t), t), t * (w(t) + x(t) + y(t) + z(t))), Eq(Derivative(z(t), t), t * (w(t) + x(t) + y(t) + z(t)) + 1), Eq(Derivative(w(t), t), t * (w(t) + x(t) + y(t) + z(t)))], 'func': [x(t), y(t), z(t), w(t)], 'order': {x(t): 1, y(t): 1, z(t): 1, w(t): 1}, 'is_linear': True, 'is_constant': False, 'is_homogeneous': False, 'is_general': True, 'func_coeff': -Matrix([[-t, -t, -t, -t], [-t, -t, -t, -t], [-t, -t, -t, -t], [-t, -t, -t, -t]]), 'commutative_antiderivative': Matrix([[t ** 2 / 2, t ** 2 / 2, t ** 2 / 2, t ** 2 / 2], [t ** 2 / 2, t ** 2 / 2, t ** 2 / 2, t ** 2 / 2], [t ** 2 / 2, t ** 2 / 2, t ** 2 / 2, t ** 2 / 2], [t ** 2 / 2, t ** 2 / 2, t ** 2 / 2, t ** 2 / 2]]), 'rhs': Matrix([[0], [0], [1], [0]]), 'type_of_equation': 'type4'}
    assert _classify_linear_system(eq5, funcs_2, t) == sol5
    t_ = symbols('t_')
    eq1 = (Eq(9 * x(t) + 7 * y(t) + 4 * Derivative(x(t), t) + Derivative(x(t), (t, 2)) + 3 * Derivative(y(t), t), 11 * exp(I * t)), Eq(3 * x(t) + 12 * y(t) + 5 * Derivative(x(t), t) + 8 * Derivative(y(t), t) + Derivative(y(t), (t, 2)), 2 * exp(I * t)))
    sol1 = {'no_of_equation': 2, 'eq': (Eq(9 * x(t) + 7 * y(t) + 4 * Derivative(x(t), t) + Derivative(x(t), (t, 2)) + 3 * Derivative(y(t), t), 11 * exp(I * t)), Eq(3 * x(t) + 12 * y(t) + 5 * Derivative(x(t), t) + 8 * Derivative(y(t), t) + Derivative(y(t), (t, 2)), 2 * exp(I * t))), 'func': [x(t), y(t)], 'order': {x(t): 2, y(t): 2}, 'is_linear': True, 'is_homogeneous': False, 'is_general': True, 'rhs': Matrix([[11 * exp(I * t)], [2 * exp(I * t)]]), 'type_of_equation': 'type0', 'is_second_order': True, 'is_higher_order': True}
    assert _classify_linear_system(eq1, funcs, t) == sol1
    eq2 = (Eq((4 * t ** 2 + 7 * t + 1) ** 2 * Derivative(x(t), (t, 2)), 5 * x(t) + 35 * y(t)), Eq((4 * t ** 2 + 7 * t + 1) ** 2 * Derivative(y(t), (t, 2)), x(t) + 9 * y(t)))
    sol2 = {'no_of_equation': 2, 'eq': (Eq((4 * t ** 2 + 7 * t + 1) ** 2 * Derivative(x(t), (t, 2)), 5 * x(t) + 35 * y(t)), Eq((4 * t ** 2 + 7 * t + 1) ** 2 * Derivative(y(t), (t, 2)), x(t) + 9 * y(t))), 'func': [x(t), y(t)], 'order': {x(t): 2, y(t): 2}, 'is_linear': True, 'is_homogeneous': True, 'is_general': True, 'type_of_equation': 'type2', 'A0': Matrix([[Rational(53, 4), 35], [1, Rational(69, 4)]]), 'g(t)': sqrt(4 * t ** 2 + 7 * t + 1), 'tau': sqrt(33) * log(t - sqrt(33) / 8 + Rational(7, 8)) / 33 - sqrt(33) * log(t + sqrt(33) / 8 + Rational(7, 8)) / 33, 'is_transformed': True, 't_': t_, 'is_second_order': True, 'is_higher_order': True}
    assert _classify_linear_system(eq2, funcs, t) == sol2
    eq3 = ((t * Derivative(x(t), t) - x(t)) * log(t) + (t * Derivative(y(t), t) - y(t)) * exp(t) + Derivative(x(t), (t, 2)), t ** 2 * (t * Derivative(x(t), t) - x(t)) + t * (t * Derivative(y(t), t) - y(t)) + Derivative(y(t), (t, 2)))
    sol3 = {'no_of_equation': 2, 'eq': ((t * Derivative(x(t), t) - x(t)) * log(t) + (t * Derivative(y(t), t) - y(t)) * exp(t) + Derivative(x(t), (t, 2)), t ** 2 * (t * Derivative(x(t), t) - x(t)) + t * (t * Derivative(y(t), t) - y(t)) + Derivative(y(t), (t, 2))), 'func': [x(t), y(t)], 'order': {x(t): 2, y(t): 2}, 'is_linear': True, 'is_homogeneous': True, 'is_general': True, 'type_of_equation': 'type1', 'A1': Matrix([[-t * log(t), -t * exp(t)], [-t ** 3, -t ** 2]]), 'is_second_order': True, 'is_higher_order': True}
    assert _classify_linear_system(eq3, funcs, t) == sol3
    eq4 = (Eq(x2, k * x(t) - l * y1), Eq(y2, l * x1 + k * y(t)))
    sol4 = {'no_of_equation': 2, 'eq': (Eq(Derivative(x(t), (t, 2)), k * x(t) - l * Derivative(y(t), t)), Eq(Derivative(y(t), (t, 2)), k * y(t) + l * Derivative(x(t), t))), 'func': [x(t), y(t)], 'order': {x(t): 2, y(t): 2}, 'is_linear': True, 'is_homogeneous': True, 'is_general': True, 'type_of_equation': 'type0', 'is_second_order': True, 'is_higher_order': True}
    assert _classify_linear_system(eq4, funcs, t) == sol4
    f, g = symbols('f g', cls=Function)
    y, t_ = symbols('y t_')
    funcs = [f(t), g(t)]
    eq1 = [Eq(Derivative(f(t), t) ** 2 - 2 * Derivative(f(t), t) + 1, 4), Eq(-y * f(t) + Derivative(g(t), t), 0)]
    sol1 = {'is_implicit': True, 'canon_eqs': [[Eq(Derivative(f(t), t), -1), Eq(Derivative(g(t), t), y * f(t))], [Eq(Derivative(f(t), t), 3), Eq(Derivative(g(t), t), y * f(t))]]}
    assert _classify_linear_system(eq1, funcs, t) == sol1
    raises(ValueError, lambda: _classify_linear_system(eq1, funcs[:1], t))
    eq2 = [Eq(Derivative(f(t), t), (2 * f(t) + g(t) + 1) / t), Eq(Derivative(g(t), t), (f(t) + 2 * g(t)) / t)]
    sol2 = {'no_of_equation': 2, 'eq': [Eq(Derivative(f(t), t), (2 * f(t) + g(t) + 1) / t), Eq(Derivative(g(t), t), (f(t) + 2 * g(t)) / t)], 'func': [f(t), g(t)], 'order': {f(t): 1, g(t): 1}, 'is_linear': True, 'is_homogeneous': False, 'is_general': True, 'rhs': Matrix([[1], [0]]), 'func_coeff': Matrix([[2, 1], [1, 2]]), 'is_constant': False, 'type_of_equation': 'type6', 't_': t_, 'tau': log(t), 'commutative_antiderivative': Matrix([[2 * log(t), log(t)], [log(t), 2 * log(t)]])}
    assert _classify_linear_system(eq2, funcs, t) == sol2
    eq3 = [Eq(Derivative(f(t), t), (2 * f(t) + g(t)) / t), Eq(Derivative(g(t), t), (f(t) + 2 * g(t)) / t)]
    sol3 = {'no_of_equation': 2, 'eq': [Eq(Derivative(f(t), t), (2 * f(t) + g(t)) / t), Eq(Derivative(g(t), t), (f(t) + 2 * g(t)) / t)], 'func': [f(t), g(t)], 'order': {f(t): 1, g(t): 1}, 'is_linear': True, 'is_homogeneous': True, 'is_general': True, 'func_coeff': Matrix([[2, 1], [1, 2]]), 'is_constant': False, 'type_of_equation': 'type5', 't_': t_, 'rhs': Matrix([[0], [0]]), 'tau': log(t), 'commutative_antiderivative': Matrix([[2 * log(t), log(t)], [log(t), 2 * log(t)]])}
    assert _classify_linear_system(eq3, funcs, t) == sol3