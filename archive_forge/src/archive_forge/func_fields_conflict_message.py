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
@classmethod
def fields_conflict_message(cls, reason_name, reason):
    return 'Fields "{}" conflict because {}. Use different aliases on the fields to fetch both if this was intentional.'.format(reason_name, cls.reason_message(reason))