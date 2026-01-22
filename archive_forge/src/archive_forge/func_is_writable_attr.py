import copy
import functools
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from ..errors import Errors
def is_writable_attr(ext):
    """Check if an extension attribute is writable.
    ext (tuple): The (default, getter, setter, method) tuple available  via
        {Doc,Span,Token}.get_extension.
    RETURNS (bool): Whether the attribute is writable.
    """
    default, method, getter, setter = ext
    if setter is not None or default is not None or all((e is None for e in ext)):
        return True
    return False