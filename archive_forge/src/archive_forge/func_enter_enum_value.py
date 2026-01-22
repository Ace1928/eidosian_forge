from typing import Any, cast
from ....error import GraphQLError
from ....language import ArgumentNode, EnumValueNode, FieldNode, ObjectFieldNode
from ....type import GraphQLInputObjectType, get_named_type, is_input_object_type
from .. import ValidationRule
def enter_enum_value(self, node: EnumValueNode, *_args: Any) -> None:
    context = self.context
    enum_value_def = context.get_enum_value()
    if enum_value_def:
        deprecation_reason = enum_value_def.deprecation_reason
        if deprecation_reason is not None:
            enum_type_def = get_named_type(context.get_input_type())
            enum_type_name = enum_type_def.name
            self.report_error(GraphQLError(f"The enum value '{enum_type_name}.{node.value}' is deprecated. {deprecation_reason}", node))