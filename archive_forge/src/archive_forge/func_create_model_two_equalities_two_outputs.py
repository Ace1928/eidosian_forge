import os
import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.pynumero.dependencies import (
from pyomo.common.dependencies.scipy import sparse as spa
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import cyipopt_available
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock
from pyomo.contrib.pynumero.interfaces.pyomo_grey_box_nlp import (
from pyomo.contrib.pynumero.interfaces.tests.compare_utils import (
import pyomo.contrib.pynumero.interfaces.tests.external_grey_box_models as ex_models
def create_model_two_equalities_two_outputs(self, external_model):
    m = pyo.ConcreteModel()
    m.hin = pyo.Var(bounds=(0, None), initialize=10)
    m.hout = pyo.Var(bounds=(0, None))
    m.egb = ExternalGreyBoxBlock()
    m.egb.set_external_model(external_model)
    m.incon = pyo.Constraint(expr=0 <= m.egb.inputs['Pin'] - 10 * m.hin)
    m.outcon = pyo.Constraint(expr=0 == m.egb.outputs['Pout'] - 10 * m.hout)
    m.egb.inputs['Pin'].value = 100
    m.egb.inputs['Pin'].setlb(50)
    m.egb.inputs['Pin'].setub(150)
    m.egb.inputs['c'].value = 2
    m.egb.inputs['c'].setlb(1)
    m.egb.inputs['c'].setub(5)
    m.egb.inputs['F'].value = 3
    m.egb.inputs['F'].setlb(1)
    m.egb.inputs['F'].setub(5)
    m.egb.inputs['P1'].value = 80
    m.egb.inputs['P1'].setlb(10)
    m.egb.inputs['P1'].setub(90)
    m.egb.inputs['P3'].value = 70
    m.egb.inputs['P3'].setlb(20)
    m.egb.inputs['P3'].setub(80)
    m.egb.outputs['P2'].value = 75
    m.egb.outputs['P2'].setlb(15)
    m.egb.outputs['P2'].setub(85)
    m.egb.outputs['Pout'].value = 50
    m.egb.outputs['Pout'].setlb(30)
    m.egb.outputs['Pout'].setub(70)
    return m