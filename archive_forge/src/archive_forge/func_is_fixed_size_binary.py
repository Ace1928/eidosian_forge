from pyarrow.lib import (is_boolean_value,  # noqa
import pyarrow.lib as lib
from pyarrow.util import doc
@doc(is_null, datatype='fixed size binary')
def is_fixed_size_binary(t):
    return t.id == lib.Type_FIXED_SIZE_BINARY