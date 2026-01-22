from typing import Collection, Dict, Optional, Tuple, Union, cast
from ..language import DirectiveLocation
from ..pyutils import inspect, merge_kwargs, natural_comparison_key
from ..type import (
def sort_types(array: Collection[GraphQLNamedType]) -> Tuple[GraphQLNamedType, ...]:
    return tuple((replace_named_type(type_) for type_ in sorted(array, key=sort_by_name_key)))