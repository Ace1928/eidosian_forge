from google.protobuf.json_format import MessageToDict
import inspect
def message_to_dict(*args, **kwargs):
    kwargs = rename_always_print_fields_with_no_presence(kwargs)
    return MessageToDict(*args, **kwargs)