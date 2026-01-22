from pyomo.core import Block
import pyomo.core.expr as EXPR
Returns a generator of all the Var objects which are used in Constraint
    expressions on the block. By default, this recurses into sub-blocks.

    Args:
        ctype: The type of component from which to get Vars, assumed to have
               an expr attribute.
        include_fixed: Whether or not to include fixed variables
        active: Whether to find Vars that appear in Constraints accessible
                via the active tree
        sort: sort method for iterating through Constraint objects
        descend_into: Ctypes to descend into when finding Constraints
        descent_order: Traversal strategy for finding the objects of type ctype
    