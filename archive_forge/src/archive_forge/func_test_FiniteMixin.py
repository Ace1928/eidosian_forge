import copy
import itertools
import logging
import pickle
from io import StringIO
from collections import namedtuple as NamedTuple
import pyomo.common.unittest as unittest
from pyomo.common import DeveloperError
from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.common.dependencies import pandas as pd, pandas_available
from pyomo.common.log import LoggingIntercept
from pyomo.core.expr import native_numeric_types, native_types
import pyomo.core.base.set as SetModule
from pyomo.core.base.indexed_component import normalize_index
from pyomo.core.base.initializer import (
from pyomo.core.base.set import (
from pyomo.environ import (
def test_FiniteMixin(self):

    class FiniteMixin(_FiniteSetMixin, _SetData):
        pass
    m = ConcreteModel()
    m.I = Set(initialize=[1])
    s = FiniteMixin(m.I)
    with self.assertRaises(DeveloperError):
        None in s
    with self.assertRaises(DeveloperError):
        self.assertFalse(s == m.I)
    with self.assertRaises(DeveloperError):
        self.assertFalse(m.I == s)
    with self.assertRaises(DeveloperError):
        self.assertTrue(s != m.I)
    with self.assertRaises(DeveloperError):
        self.assertTrue(m.I != s)
    with self.assertRaises(DeveloperError):
        str(s)
    with self.assertRaises(DeveloperError):
        s.dimen
    with self.assertRaises(DeveloperError):
        s.domain
    self.assertTrue(s.isfinite())
    self.assertFalse(s.isordered())
    range_iter = s.ranges()
    with self.assertRaises(DeveloperError):
        list(range_iter)
    with self.assertRaises(DeveloperError):
        s.isdisjoint(m.I)
    with self.assertRaises(DeveloperError):
        m.I.isdisjoint(s)
    with self.assertRaises(DeveloperError):
        s.issuperset(m.I)
    with self.assertRaises(DeveloperError):
        self.assertFalse(m.I.issuperset(s))
    with self.assertRaises(DeveloperError):
        self.assertFalse(s.issubset(m.I))
    with self.assertRaises(DeveloperError):
        m.I.issubset(s)
    self.assertIs(type(s.union(m.I)), SetUnion_FiniteSet)
    self.assertIs(type(m.I.union(s)), SetUnion_FiniteSet)
    self.assertIs(type(s.intersection(m.I)), SetIntersection_OrderedSet)
    self.assertIs(type(m.I.intersection(s)), SetIntersection_OrderedSet)
    self.assertIs(type(s.difference(m.I)), SetDifference_FiniteSet)
    self.assertIs(type(m.I.difference(s)), SetDifference_OrderedSet)
    self.assertIs(type(s.symmetric_difference(m.I)), SetSymmetricDifference_FiniteSet)
    self.assertIs(type(m.I.symmetric_difference(s)), SetSymmetricDifference_FiniteSet)
    self.assertIs(type(s.cross(m.I)), SetProduct_FiniteSet)
    self.assertIs(type(m.I.cross(s)), SetProduct_FiniteSet)
    self.assertIs(type(s | m.I), SetUnion_FiniteSet)
    self.assertIs(type(m.I | s), SetUnion_FiniteSet)
    self.assertIs(type(s & m.I), SetIntersection_OrderedSet)
    self.assertIs(type(m.I & s), SetIntersection_OrderedSet)
    self.assertIs(type(s - m.I), SetDifference_FiniteSet)
    self.assertIs(type(m.I - s), SetDifference_OrderedSet)
    self.assertIs(type(s ^ m.I), SetSymmetricDifference_FiniteSet)
    self.assertIs(type(m.I ^ s), SetSymmetricDifference_FiniteSet)
    self.assertIs(type(s * m.I), SetProduct_FiniteSet)
    self.assertIs(type(m.I * s), SetProduct_FiniteSet)
    with self.assertRaises(DeveloperError):
        self.assertFalse(s < m.I)
    with self.assertRaises(DeveloperError):
        self.assertFalse(m.I < s)
    with self.assertRaises(DeveloperError):
        self.assertFalse(s > m.I)
    with self.assertRaises(DeveloperError):
        self.assertFalse(m.I > s)
    with self.assertRaises(DeveloperError):
        len(s)
    with self.assertRaises(DeveloperError):
        iter(s)
    with self.assertRaises(DeveloperError):
        reversed(s)
    with self.assertRaises(DeveloperError):
        s.data()
    with self.assertRaises(DeveloperError):
        s.ordered_data()
    with self.assertRaises(DeveloperError):
        s.sorted_data()
    self.assertEqual(s.bounds(), (None, None))