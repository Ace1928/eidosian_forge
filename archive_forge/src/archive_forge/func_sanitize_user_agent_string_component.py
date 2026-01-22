import os
import platform
from copy import copy
from string import ascii_letters, digits
from typing import NamedTuple, Optional
from botocore import __version__ as botocore_version
from botocore.compat import HAS_CRT
def sanitize_user_agent_string_component(raw_str, allow_hash):
    """Replaces all not allowed characters in the string with a dash ("-").

    Allowed characters are ASCII alphanumerics and ``!$%&'*+-.^_`|~``. If
    ``allow_hash`` is ``True``, "#"``" is also allowed.

    :type raw_str: str
    :param raw_str: The input string to be sanitized.

    :type allow_hash: bool
    :param allow_hash: Whether "#" is considered an allowed character.
    """
    return ''.join((c if c in _USERAGENT_ALLOWED_CHARACTERS or (allow_hash and c == '#') else '-' for c in raw_str))