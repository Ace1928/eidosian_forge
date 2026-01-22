from google.protobuf import duration_pb2
from google.protobuf import descriptor
from google.protobuf import message
def math_opt_normalize_proto(protobuf_message: message.Message) -> None:
    """Clears all non-duration submessages that are not in one_ofs.

    A message is considered `empty` if:
      * every non-optional scalar fields has its default value,
      * every optional scalar field is unset,
      * every repeated/map fields is empty
      * every oneof is unset,
      * every duration field is unset
      * all other message fields (singular, not oneof, not duration) are `empty`.
    This function clears all `empty` fields from `message`.

    This is useful for testing.

    Args:
      protobuf_message: The Message object to clear.
    """
    for field, value in protobuf_message.ListFields():
        if field.type != field.TYPE_MESSAGE:
            continue
        if field.label == field.LABEL_REPEATED:
            if field.message_type.has_options and field.message_type.GetOptions().map_entry:
                if field.message_type.fields_by_number[2].type == descriptor.FieldDescriptor.TYPE_MESSAGE:
                    for item in value.values():
                        math_opt_normalize_proto(item)
            else:
                for item in value:
                    math_opt_normalize_proto(item)
            continue
        math_opt_normalize_proto(value)
        if not value.ListFields() and field.message_type != duration_pb2.Duration.DESCRIPTOR and (field.containing_oneof is None):
            protobuf_message.ClearField(field.name)