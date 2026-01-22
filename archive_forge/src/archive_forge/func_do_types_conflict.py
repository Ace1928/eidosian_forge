import itertools
from collections import OrderedDict
from ...error import GraphQLError
from ...language import ast
from ...language.printer import print_ast
from ...pyutils.pair_set import PairSet
from ...type.definition import (GraphQLInterfaceType, GraphQLList,
from ...utils.type_comparators import is_equal_type
from ...utils.type_from_ast import type_from_ast
from .base import ValidationRule
def do_types_conflict(type1, type2):
    if isinstance(type1, GraphQLList):
        if isinstance(type2, GraphQLList):
            return do_types_conflict(type1.of_type, type2.of_type)
        return True
    if isinstance(type2, GraphQLList):
        if isinstance(type1, GraphQLList):
            return do_types_conflict(type1.of_type, type2.of_type)
        return True
    if isinstance(type1, GraphQLNonNull):
        if isinstance(type2, GraphQLNonNull):
            return do_types_conflict(type1.of_type, type2.of_type)
        return True
    if isinstance(type2, GraphQLNonNull):
        if isinstance(type1, GraphQLNonNull):
            return do_types_conflict(type1.of_type, type2.of_type)
        return True
    if is_leaf_type(type1) or is_leaf_type(type2):
        return type1 != type2
    return False