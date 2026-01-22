import enum
from pyomo.common.dependencies import attempt_import
from pyomo.common.numeric_types import native_types
from pyomo.core.pyomoobject import PyomoObject
from pyomo.core.expr.expr_common import OperatorAssociativity
def potentially_variable_base_class(self):
    cls = list(self.__class__.__bases__)
    cls.remove(NPV_Mixin)
    assert len(cls) == 1
    return cls[0]