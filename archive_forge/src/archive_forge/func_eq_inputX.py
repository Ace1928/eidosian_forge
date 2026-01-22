from pyomo.environ import (
from pyomo.common.collections import ComponentMap
from pyomo.contrib.mindtpy.tests.MINLP_simple_grey_box import (
def eq_inputX(m):
    return m.X[i] == m.my_block.egb.inputs['X' + str(i)]