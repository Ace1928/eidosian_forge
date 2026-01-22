from marshmallow.exceptions import SCHEMA
def store_error(self, messages, field_name=SCHEMA, index=None):
    if field_name != SCHEMA or not isinstance(messages, dict):
        messages = {field_name: messages}
    if index is not None:
        messages = {index: messages}
    self.errors = merge_errors(self.errors, messages)