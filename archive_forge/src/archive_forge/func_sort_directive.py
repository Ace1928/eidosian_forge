from typing import Collection, Dict, Optional, Tuple, Union, cast
from ..language import DirectiveLocation
from ..pyutils import inspect, merge_kwargs, natural_comparison_key
from ..type import (
def sort_directive(directive: GraphQLDirective) -> GraphQLDirective:
    return GraphQLDirective(**merge_kwargs(directive.to_kwargs(), locations=sorted(directive.locations, key=sort_by_name_key), args=sort_args(directive.args)))