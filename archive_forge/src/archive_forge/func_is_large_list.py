from pyarrow.lib import (is_boolean_value,  # noqa
import pyarrow.lib as lib
from pyarrow.util import doc
@doc(is_null, datatype='large list')
def is_large_list(t):
    return t.id == lib.Type_LARGE_LIST