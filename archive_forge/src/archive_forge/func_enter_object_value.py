from typing import cast, Any
from ...error import GraphQLError
from ...language import (
from ...pyutils import did_you_mean, suggestion_list, Undefined
from ...type import (
from . import ValidationRule
def enter_object_value(self, node: ObjectValueNode, *_args: Any) -> VisitorAction:
    type_ = get_named_type(self.context.get_input_type())
    if not is_input_object_type(type_):
        self.is_valid_value_node(node)
        return SKIP
    type_ = cast(GraphQLInputObjectType, type_)
    field_node_map = {field.name.value: field for field in node.fields}
    for field_name, field_def in type_.fields.items():
        field_node = field_node_map.get(field_name)
        if not field_node and is_required_input_field(field_def):
            field_type = field_def.type
            self.report_error(GraphQLError(f"Field '{type_.name}.{field_name}' of required type '{field_type}' was not provided.", node))
    return None