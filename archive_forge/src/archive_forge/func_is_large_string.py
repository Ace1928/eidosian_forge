from pyarrow.lib import (is_boolean_value,  # noqa
import pyarrow.lib as lib
from pyarrow.util import doc
@doc(is_null, datatype='large string (utf8 unicode)')
def is_large_string(t):
    return t.id == lib.Type_LARGE_STRING