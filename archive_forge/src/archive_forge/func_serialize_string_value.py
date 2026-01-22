def serialize_string_value(value):
    return ''.join(('\\"' if c == '"' else '\\\\' if c == '\\' else '\\A ' if c == '\n' else '\\D ' if c == '\r' else '\\C ' if c == '\x0c' else c for c in value))