import functools
import numpy as np
import pyarrow as pa
from modin.core.dataframe.base.interchange.dataframe_protocol.utils import (
def raise_copy_alert_if_materialize(fn):
    """
    Decorate ``HdkProtocolDataframe`` method with a check raising a copy-alert if it's impossible to retrieve the data in zero-copy way.

    Parameters
    ----------
    fn : callable
        ``HdkProtocolDataframe`` method.

    Returns
    -------
    callable
    """

    @functools.wraps(fn)
    def method(self, *args, **kwargs):
        if not self._allow_copy and (not self._is_zero_copy_possible):
            raise_copy_alert()
        return fn(self, *args, **kwargs)
    return method