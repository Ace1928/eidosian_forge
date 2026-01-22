from sympy.core.basic import Basic
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.parameters import evaluate
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.geometry import Line, Point, Point2D, Point3D, Line3D, Plane
from sympy.geometry.entity import rotate, scale, translate, GeometryEntity
from sympy.matrices import Matrix
from sympy.utilities.iterables import subsets, permutations, cartes
from sympy.utilities.misc import Undecidable
from sympy.testing.pytest import raises, warns
def test__normalize_dimension():
    assert Point._normalize_dimension(Point(1, 2), Point(3, 4)) == [Point(1, 2), Point(3, 4)]
    assert Point._normalize_dimension(Point(1, 2), Point(3, 4, 0), on_morph='ignore') == [Point(1, 2, 0), Point(3, 4, 0)]