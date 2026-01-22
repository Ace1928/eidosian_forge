from typing import Collection, Dict, Optional, Tuple, Union, cast
from ..language import DirectiveLocation
from ..pyutils import inspect, merge_kwargs, natural_comparison_key
from ..type import (
def replace_maybe_type(maybe_type: Optional[GraphQLNamedType]) -> Optional[GraphQLNamedType]:
    return maybe_type and replace_named_type(maybe_type)