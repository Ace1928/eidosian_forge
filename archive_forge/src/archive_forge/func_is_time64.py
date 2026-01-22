from pyarrow.lib import (is_boolean_value,  # noqa
import pyarrow.lib as lib
from pyarrow.util import doc
@doc(is_null, datatype='time64')
def is_time64(t):
    return t.id == lib.Type_TIME64