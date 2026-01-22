from torch.fx.experimental.migrate_gradual_types.constraint import Conj, Disj, T, F, BinConstraintT, BVar, is_bool_expr
from torch.fx.experimental.migrate_gradual_types.constraint import BinConstraintD, TVar, DVar
from torch.fx.experimental.migrate_gradual_types.constraint import Prod, is_algebraic_expression, is_dim
from torch.fx.experimental.migrate_gradual_types.constraint_generator import ConstraintGenerator
from torch.fx.experimental.migrate_gradual_types.constraint_transformation import transform_constraint
from torch.fx.experimental.migrate_gradual_types.operation import op_add, op_eq, op_neq, op_gt, op_lt
from torch.fx.experimental.migrate_gradual_types.operation import op_leq, op_sub, op_div, op_mul, op_mod
from torch.fx.tensor_type import TensorType, Dyn
def transform_all_constraints(traced, counter=0):
    """
        Given a trace, generates constraints and transforms them to z3 format

        """
    dimension_dict = {}
    generator = ConstraintGenerator(traced)
    new_constraints, counter = generator.generate_constraints(counter)
    new_constraints, counter = iterate_till_fixed_point(new_constraints, counter)
    transformed, counter = transform_to_z3(new_constraints, counter, dimension_dict)
    return transformed