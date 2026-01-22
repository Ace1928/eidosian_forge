from ...error import GraphQLError
from ...utils.type_comparators import do_types_overlap
from ...utils.type_from_ast import type_from_ast
from .base import ValidationRule
@staticmethod
def get_fragment_type(context, name):
    frag = context.get_fragment(name)
    return frag and type_from_ast(context.get_schema(), frag.type_condition)