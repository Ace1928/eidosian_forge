import re
from .checks import check_data, check_msgdict, check_value
from .decode import decode_message
from .encode import encode_message
from .specs import REALTIME_TYPES, SPEC_BY_TYPE, make_msgdict
from .strings import msg2str, str2msg
def parse_string_stream(stream):
    """Parse a stream of messages and yield (message, error_message)

    stream can be any iterable that generates text strings, where each
    string is a string encoded message.

    If a string can be parsed, (message, None) is returned. If it
    can't be parsed, (None, error_message) is returned. The error
    message contains the line number where the error occurred.
    """
    line_number = 1
    for line in stream:
        try:
            line = line.split('#')[0].strip()
            if line:
                yield (parse_string(line), None)
        except ValueError as exception:
            error_message = 'line {line_number}: {msg}'.format(line_number=line_number, msg=exception.args[0])
            yield (None, error_message)
        line_number += 1