import re
import sys
import urllib   # For urllib.parse.unquote
from string import hexdigits
from operator import itemgetter
from email import _encoded_words as _ew
from email import errors
from email import utils
def parse_message_id(value):
    """message-id      =   "Message-ID:" msg-id CRLF
    """
    message_id = MessageID()
    try:
        token, value = get_msg_id(value)
        message_id.append(token)
    except errors.HeaderParseError as ex:
        token = get_unstructured(value)
        message_id = InvalidMessageID(token)
        message_id.defects.append(errors.InvalidHeaderDefect('Invalid msg-id: {!r}'.format(ex)))
    else:
        if value:
            message_id.defects.append(errors.InvalidHeaderDefect('Unexpected {!r}'.format(value)))
    return message_id