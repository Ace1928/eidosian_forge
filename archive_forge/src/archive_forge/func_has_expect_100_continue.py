import re
from typing import AnyStr, cast, List, overload, Sequence, Tuple, TYPE_CHECKING, Union
from ._abnf import field_name, field_value
from ._util import bytesify, LocalProtocolError, validate
def has_expect_100_continue(request: 'Request') -> bool:
    if request.http_version < b'1.1':
        return False
    expect = get_comma_header(request.headers, b'expect')
    return b'100-continue' in expect