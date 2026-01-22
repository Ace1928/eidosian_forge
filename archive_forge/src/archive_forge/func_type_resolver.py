import sys
from collections.abc import Iterable
from functools import partial
from wandb_promise import Promise, is_thenable
from ...error import GraphQLError, GraphQLLocatedError
from ...type import (GraphQLEnumType, GraphQLInterfaceType, GraphQLList,
from ..base import default_resolve_fn
from ...execution import executor
from .utils import imap, normal_map
def type_resolver(return_type, resolver, fragment=None, exe_context=None, info=None, catch_error=False):
    if isinstance(return_type, GraphQLNonNull):
        return type_resolver_non_null(return_type, resolver, fragment, exe_context, info)
    if isinstance(return_type, (GraphQLScalarType, GraphQLEnumType)):
        return type_resolver_leaf(return_type, resolver, exe_context, info, catch_error)
    if isinstance(return_type, GraphQLList):
        return type_resolver_list(return_type, resolver, fragment, exe_context, info, catch_error)
    if isinstance(return_type, GraphQLObjectType):
        assert fragment and fragment.type == return_type, 'Fragment and return_type dont match'
        return type_resolver_fragment(return_type, resolver, fragment, exe_context, info, catch_error)
    if isinstance(return_type, (GraphQLInterfaceType, GraphQLUnionType)):
        assert fragment, 'You need to pass a fragment to resolve a Interface or Union'
        return type_resolver_fragment(return_type, resolver, fragment, exe_context, info, catch_error)
    raise Exception('The resolver have to be created for a fragment')