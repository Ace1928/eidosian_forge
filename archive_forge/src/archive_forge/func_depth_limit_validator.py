from typing import Callable, Dict, List, Optional, Union, Tuple
from graphql import GraphQLError
from graphql.validation import ValidationContext, ValidationRule
from graphql.language import (
from ..utils.is_introspection_key import is_introspection_key
def depth_limit_validator(max_depth: int, ignore: Optional[List[IgnoreType]]=None, callback: Optional[Callable[[Dict[str, int]], None]]=None):

    class DepthLimitValidator(ValidationRule):

        def __init__(self, validation_context: ValidationContext):
            document = validation_context.document
            definitions = document.definitions
            fragments = get_fragments(definitions)
            queries = get_queries_and_mutations(definitions)
            query_depths = {}
            for name in queries:
                query_depths[name] = determine_depth(node=queries[name], fragments=fragments, depth_so_far=0, max_depth=max_depth, context=validation_context, operation_name=name, ignore=ignore)
            if callable(callback):
                callback(query_depths)
            super().__init__(validation_context)
    return DepthLimitValidator