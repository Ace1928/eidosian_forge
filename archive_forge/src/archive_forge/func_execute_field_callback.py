import functools
from wandb_promise import Promise, is_thenable, promise_for_dict
from ...pyutils.cached_property import cached_property
from ...pyutils.default_ordered_dict import DefaultOrderedDict
from ...type import (GraphQLInterfaceType, GraphQLList, GraphQLNonNull,
from ..base import ResolveInfo, Undefined, collect_fields, get_field_def
from ..values import get_argument_values
from ...error import GraphQLError
def execute_field_callback(results, resolver):
    response_name, field_resolver = resolver
    result = field_resolver.execute(root)
    if result is Undefined:
        return results
    if is_thenable(result):

        def collect_result(resolved_result):
            results[response_name] = resolved_result
            return results
        return result.then(collect_result)
    results[response_name] = result
    return results