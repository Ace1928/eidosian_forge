import logging
import math
import operator
import sys
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.core.expr.compare import (
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from pyomo.core.expr.visitor import clone_expression
from pyomo.environ import ConcreteModel, Param, Var, BooleanVar
from pyomo.gdp import Disjunct
def test_categorize_arg_type(self):

    class CustomAsNumeric(NumericValue):

        def __init__(self, val):
            self._value = val

        def is_numeric_type(self):
            return False

        def as_numeric(self):
            return self._value

    class CustomNumeric(NumericValue):

        def is_potentially_variable(self):
            return True
    saved_known_arg_types = dict(_known_arg_types)
    saved_native_numeric_types = set(native_numeric_types)
    saved_native_types = set(native_types)
    try:
        _known_arg_types.clear()
        native_types.discard(int)
        native_numeric_types.discard(int)
        self.assertIs(_categorize_arg_type(10), ARG_TYPE.NATIVE)
        self.assertIn(int, _known_arg_types)
        self.assertIn(int, native_numeric_types)
        _known_arg_types.clear()
        self.assertIs(_categorize_arg_type(10), ARG_TYPE.NATIVE)
        self.assertIs(_categorize_arg_type(10), ARG_TYPE.NATIVE)
        self.assertIs(_categorize_arg_type(self.invalid), ARG_TYPE.INVALID)
        self.assertIs(_categorize_arg_type(self.asbinary), ARG_TYPE.ASNUMERIC)
        self.assertIs(_categorize_arg_type(self.zero), ARG_TYPE.NATIVE)
        self.assertIs(_categorize_arg_type(self.one), ARG_TYPE.NATIVE)
        self.assertIs(_categorize_arg_type(self.native), ARG_TYPE.NATIVE)
        self.assertIs(_categorize_arg_type(self.npv), ARG_TYPE.NPV)
        self.assertIs(_categorize_arg_type(self.param), ARG_TYPE.PARAM)
        self.assertIs(_categorize_arg_type(self.param0), ARG_TYPE.PARAM)
        self.assertIs(_categorize_arg_type(self.param1), ARG_TYPE.PARAM)
        self.assertIs(_categorize_arg_type(self.param_mut), ARG_TYPE.PARAM)
        self.assertIs(_categorize_arg_type(self.var), ARG_TYPE.VAR)
        self.assertIs(_categorize_arg_type(self.mon_native), ARG_TYPE.MONOMIAL)
        self.assertIs(_categorize_arg_type(self.mon_param), ARG_TYPE.MONOMIAL)
        self.assertIs(_categorize_arg_type(self.mon_npv), ARG_TYPE.MONOMIAL)
        self.assertIs(_categorize_arg_type(self.linear), ARG_TYPE.LINEAR)
        self.assertIs(_categorize_arg_type(self.sum), ARG_TYPE.SUM)
        self.assertIs(_categorize_arg_type(self.other), ARG_TYPE.OTHER)
        self.assertIs(_categorize_arg_type(self.mutable_l0), ARG_TYPE.MUTABLE)
        self.assertIs(_categorize_arg_type(self.mutable_l1), ARG_TYPE.MUTABLE)
        self.assertIs(_categorize_arg_type(self.mutable_l2), ARG_TYPE.MUTABLE)
        self.assertIs(_categorize_arg_type(self.mutable_l3), ARG_TYPE.MUTABLE)
        self.assertIs(_categorize_arg_type(self.m), ARG_TYPE.INVALID)
        self.assertIs(_categorize_arg_type(CustomAsNumeric(10)), ARG_TYPE.ASNUMERIC)
        self.assertIs(_categorize_arg_type(CustomNumeric()), ARG_TYPE.OTHER)
    finally:
        _known_arg_types.clear()
        _known_arg_types.update(saved_known_arg_types)
        native_numeric_types.clear()
        native_numeric_types.update(saved_native_numeric_types)
        native_types.clear()
        native_types.update(saved_native_types)