from ..lazy_import import lazy_import
import time
from breezy import (
from .. import lazy_regex
def gen_file_id(name):
    """Return new file id for the basename 'name'.

    The uniqueness is supplied from _next_id_suffix.
    """
    if isinstance(name, str):
        name = name.encode('ascii', 'replace')
    ascii_word_only = _file_id_chars_re.sub(b'', name.lower())
    short_no_dots = ascii_word_only.lstrip(b'.')[:20]
    return short_no_dots + _next_id_suffix()