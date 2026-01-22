from pyarrow.lib import (is_boolean_value,  # noqa
import pyarrow.lib as lib
from pyarrow.util import doc
@doc(is_null, datatype='float32 (single precision)')
def is_float32(t):
    return t.id == lib.Type_FLOAT