from typing import cast, Any
from ...error import GraphQLError
from ...language import (
from ...pyutils import did_you_mean, suggestion_list, Undefined
from ...type import (
from . import ValidationRule
def enter_float_value(self, node: FloatValueNode, *_args: Any) -> None:
    self.is_valid_value_node(node)