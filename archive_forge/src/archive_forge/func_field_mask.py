import collections
import collections.abc
import copy
import inspect
from cloudsdk.google.protobuf import field_mask_pb2
from cloudsdk.google.protobuf import message
from cloudsdk.google.protobuf import wrappers_pb2
def field_mask(original, modified):
    """Create a field mask by comparing two messages.

    Args:
        original (~google.protobuf.message.Message): the original message.
            If set to None, this field will be interpreted as an empty
            message.
        modified (~google.protobuf.message.Message): the modified message.
            If set to None, this field will be interpreted as an empty
            message.

    Returns:
        google.protobuf.field_mask_pb2.FieldMask: field mask that contains
        the list of field names that have different values between the two
        messages. If the messages are equivalent, then the field mask is empty.

    Raises:
        ValueError: If the ``original`` or ``modified`` are not the same type.
    """
    if original is None and modified is None:
        return field_mask_pb2.FieldMask()
    if original is None and modified is not None:
        original = copy.deepcopy(modified)
        original.Clear()
    if modified is None and original is not None:
        modified = copy.deepcopy(original)
        modified.Clear()
    if not isinstance(original, type(modified)):
        raise ValueError('expected that both original and modified should be of the same type, received "{!r}" and "{!r}".'.format(type(original), type(modified)))
    return field_mask_pb2.FieldMask(paths=_field_mask_helper(original, modified))