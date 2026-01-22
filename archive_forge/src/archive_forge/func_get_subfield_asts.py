import functools
from wandb_promise import Promise, is_thenable, promise_for_dict
from ...pyutils.cached_property import cached_property
from ...pyutils.default_ordered_dict import DefaultOrderedDict
from ...type import (GraphQLInterfaceType, GraphQLList, GraphQLNonNull,
from ..base import ResolveInfo, Undefined, collect_fields, get_field_def
from ..values import get_argument_values
from ...error import GraphQLError
def get_subfield_asts(context, return_type, field_asts):
    subfield_asts = DefaultOrderedDict(list)
    visited_fragment_names = set()
    for field_ast in field_asts:
        selection_set = field_ast.selection_set
        if selection_set:
            subfield_asts = collect_fields(context, return_type, selection_set, subfield_asts, visited_fragment_names)
    return subfield_asts