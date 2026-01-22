from torch.fx.experimental.graph_gradual_typechecker import Refine
from torch.fx.tensor_type import TensorType
from torch.fx.experimental.unification import Var, unify  # type: ignore[attr-defined]
def convert_eq(list_of_eq):
    """
    Convert equality constraints in the right format
    to be used by unification library.
    """
    lhs = []
    rhs = []
    for eq in list_of_eq:
        lhs.append(eq.lhs)
        rhs.append(eq.rhs)
    return (tuple(lhs), tuple(rhs))