import re
from typing import Dict, Union
from xml.etree.ElementTree import (Element, ElementTree, ParseError,
from .. import errors, lazy_regex
from . import inventory, serializer
def escape_invalid_chars(message):
    """Escape the XML-invalid characters in a commit message.

    :param message: Commit message to escape
    :return: tuple with escaped message and number of characters escaped
    """
    if message is None:
        return (None, 0)
    return re.subn('[^\t\n\r -\ud7ff\ue000-�]+', lambda match: match.group(0).encode('unicode_escape').decode('ascii'), message)