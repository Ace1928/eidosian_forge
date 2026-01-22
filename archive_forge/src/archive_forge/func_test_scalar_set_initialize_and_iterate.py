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
def test_scalar_set_initialize_and_iterate(self):
    m = ConcreteModel()
    m.I = Set()
    self.assertEqual(list(m.I), [])
    self.assertEqual(list(reversed(m.I)), [])
    self.assertEqual(m.I.data(), ())
    self.assertEqual(m.I.dimen, UnknownSetDimen)
    m = ConcreteModel()
    with self.assertRaisesRegex(KeyError, "Cannot treat the scalar component 'I' as an indexed component"):
        m.I = Set(initialize={1: (1, 3, 2, 4)})
    m = ConcreteModel()
    m.I = Set(initialize=(1, 3, 2, 4))
    self.assertTrue(m.I._init_values.constant())
    self.assertEqual(list(m.I), [1, 3, 2, 4])
    self.assertEqual(list(reversed(m.I)), [4, 2, 3, 1])
    self.assertEqual(m.I.data(), (1, 3, 2, 4))
    self.assertEqual(m.I.dimen, 1)
    m = ConcreteModel()
    with self.assertRaisesRegex(ValueError, 'Set rule or initializer returned None'):
        m.I = Set(initialize=lambda m: None, dimen=2)
    self.assertTrue(m.I._init_values.constant())
    self.assertEqual(list(m.I), [])
    self.assertEqual(list(reversed(m.I)), [])
    self.assertEqual(m.I.data(), ())
    self.assertIs(m.I.dimen, 2)

    def I_init(m):
        yield 1
        yield 3
        yield 2
        yield 4
    m = ConcreteModel()
    m.I = Set(initialize=I_init)
    self.assertEqual(list(m.I), [1, 3, 2, 4])
    self.assertEqual(list(reversed(m.I)), [4, 2, 3, 1])
    self.assertEqual(m.I.data(), (1, 3, 2, 4))
    self.assertEqual(m.I.dimen, 1)
    m = ConcreteModel()
    m.I = Set(initialize={None: (1, 3, 2, 4)})
    self.assertEqual(list(m.I), [1, 3, 2, 4])
    self.assertEqual(list(reversed(m.I)), [4, 2, 3, 1])
    self.assertEqual(m.I.data(), (1, 3, 2, 4))
    self.assertEqual(m.I.dimen, 1)
    output = StringIO()
    with LoggingIntercept(output, 'pyomo.core'):
        m = ConcreteModel()
        m.I = Set(initialize={1, 3, 2, 4})
        ref = 'Initializing ordered Set I with a fundamentally unordered data source (type: set).'
        self.assertIn(ref, output.getvalue())
    self.assertEqual(m.I.sorted_data(), (1, 2, 3, 4))
    self.assertEqual(list(reversed(list(m.I))), list(reversed(m.I)))
    self.assertEqual(list(reversed(m.I.data())), list(reversed(m.I)))
    self.assertEqual(m.I.dimen, 1)
    output = StringIO()
    with LoggingIntercept(output, 'pyomo.core'):
        m = ConcreteModel()
        m.I = Set(initialize={1, 3, 2, 4}, ordered=False)
        self.assertEqual(output.getvalue(), '')
    self.assertEqual(sorted(list(m.I)), [1, 2, 3, 4])
    self.assertEqual(list(reversed(list(m.I))), list(reversed(m.I)))
    self.assertEqual(list(reversed(m.I.data())), list(reversed(m.I)))
    self.assertEqual(m.I.dimen, 1)
    m = ConcreteModel()
    m.I = Set(initialize=[1, 3, 2, 4], ordered=Set.SortedOrder)
    self.assertEqual(list(m.I), [1, 2, 3, 4])
    self.assertEqual(list(reversed(m.I)), [4, 3, 2, 1])
    self.assertEqual(m.I.data(), (1, 2, 3, 4))
    self.assertEqual(m.I.dimen, 1)
    with self.assertRaisesRegex(TypeError, "Set 'ordered' argument is not valid \\(must be one of {False, True, <function>, Set.InsertionOrder, Set.SortedOrder}\\)"):
        m = ConcreteModel()
        m.I = Set(initialize=[1, 3, 2, 4], ordered=Set)
    m = ConcreteModel()
    m.I = Set(initialize=[1, 3, 2, 4], ordered=lambda x: reversed(sorted(x)))
    self.assertEqual(list(m.I), [4, 3, 2, 1])
    self.assertEqual(list(reversed(m.I)), [1, 2, 3, 4])
    self.assertEqual(m.I.data(), (4, 3, 2, 1))
    self.assertEqual(m.I.dimen, 1)
    output = StringIO()
    with LoggingIntercept(output, 'pyomo.core'):
        with self.assertRaisesRegex(TypeError, "'int' object is not iterable"):
            m = ConcreteModel()
            m.I = Set(initialize=5)
        ref = 'Initializer for Set I returned non-iterable object of type int.'
        self.assertIn(ref, output.getvalue())