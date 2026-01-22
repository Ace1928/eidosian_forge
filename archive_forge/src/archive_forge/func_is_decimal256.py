from pyarrow.lib import (is_boolean_value,  # noqa
import pyarrow.lib as lib
from pyarrow.util import doc
@doc(is_null, datatype='decimal256')
def is_decimal256(t):
    return t.id == lib.Type_DECIMAL256