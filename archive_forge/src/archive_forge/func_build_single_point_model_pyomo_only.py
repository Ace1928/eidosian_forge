import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.external_grey_box import (
import scipy.sparse as spa
import numpy as np
import math
def build_single_point_model_pyomo_only(m):
    m.Cp_h = 2131
    m.Cp_c = 4178
    m.Fh = 0.1
    m.Fc = 0.2
    m.Th_in = pyo.Var(initialize=100, bounds=(10, None))
    m.Th_out = pyo.Var(initialize=50, bounds=(10, None))
    m.Tc_in = pyo.Var(initialize=30, bounds=(10, None))
    m.Tc_out = pyo.Var(initialize=50, bounds=(10, None))
    m.UA = pyo.Var(initialize=100)
    m.Q = pyo.Var(initialize=10000, bounds=(0, None))
    m.lmtd = pyo.Var(initialize=20, bounds=(0, None))
    m.dt1 = pyo.Var(initialize=20, bounds=(0, None))
    m.dt2 = pyo.Var(initialize=20, bounds=(0, None))
    m.dt1_con = pyo.Constraint(expr=m.dt1 == m.Th_in - m.Tc_out)
    m.dt2_con = pyo.Constraint(expr=m.dt2 == m.Th_out - m.Tc_in)
    m.lmtd_con = pyo.Constraint(expr=m.lmtd * pyo.log(m.dt2 / m.dt1) == m.dt2 - m.dt1)
    m.ua_con = pyo.Constraint(expr=m.Q == m.UA * m.lmtd)
    m.Qh_con = pyo.Constraint(expr=m.Q == m.Fh * m.Cp_h * (m.Th_in - m.Th_out))
    m.Qc_con = pyo.Constraint(expr=m.Q == m.Fc * m.Cp_c * (m.Tc_out - m.Tc_in))