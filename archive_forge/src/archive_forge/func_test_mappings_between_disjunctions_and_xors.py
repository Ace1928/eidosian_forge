from pyomo.common.dependencies import dill_available
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
import logging
from pyomo.environ import (
from pyomo.core.expr.compare import (
import pyomo.core.expr as EXPR
from pyomo.core.base import constraint
from pyomo.repn import generate_standard_repn
from pyomo.repn.linear import LinearRepnVisitor
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
import pyomo.gdp.tests.models as models
import pyomo.gdp.tests.common_tests as ct
import random
from io import StringIO
import os
from os.path import abspath, dirname, join
from filecmp import cmp
def test_mappings_between_disjunctions_and_xors(self):
    m = models.makeNestedDisjunctions()
    transform = TransformationFactory('gdp.hull')
    transform.apply_to(m)
    transBlock = m.component('_pyomo_gdp_hull_reformulation')
    disjunctionPairs = [(m.disjunction, transBlock.disjunction_xor), (m.disjunct[1].innerdisjunction[0], m.disjunct[1].innerdisjunction[0].algebraic_constraint.parent_block().innerdisjunction_xor[0]), (m.simpledisjunct.innerdisjunction, m.simpledisjunct.innerdisjunction.algebraic_constraint.parent_block().innerdisjunction_xor)]
    for disjunction, xor in disjunctionPairs:
        self.assertIs(disjunction.algebraic_constraint, xor)
        self.assertIs(transform.get_src_disjunction(xor), disjunction)