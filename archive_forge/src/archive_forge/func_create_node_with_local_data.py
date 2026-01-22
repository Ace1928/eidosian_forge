from pyomo.common.autoslots import AutoSlots
from pyomo.core.expr.numeric_expr import NumericExpression
from weakref import ref as weakref_ref
def create_node_with_local_data(self, args):
    return self.__class__(args, pw_linear_function=self.pw_linear_function)