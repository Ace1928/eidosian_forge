import abc
import copy
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.utilities import performance_utils as pu
from cvxpy.utilities.deterministic import unique_list
def tree_copy(self, id_objects=None):
    new_args = []
    for arg in self.args:
        if isinstance(arg, list):
            arg_list = [elem.tree_copy(id_objects) for elem in arg]
            new_args.append(arg_list)
        else:
            new_args.append(arg.tree_copy(id_objects))
    return self.copy(args=new_args, id_objects=id_objects)