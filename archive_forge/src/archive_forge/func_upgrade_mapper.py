import numpy as np
import numpy.core.numeric as nx
from numpy.compat import asbytes, asunicode
@classmethod
def upgrade_mapper(cls, func, default=None):
    """
        Upgrade the mapper of a StringConverter by adding a new function and
        its corresponding default.

        The input function (or sequence of functions) and its associated
        default value (if any) is inserted in penultimate position of the
        mapper.  The corresponding type is estimated from the dtype of the
        default value.

        Parameters
        ----------
        func : var
            Function, or sequence of functions

        Examples
        --------
        >>> import dateutil.parser
        >>> import datetime
        >>> dateparser = dateutil.parser.parse
        >>> defaultdate = datetime.date(2000, 1, 1)
        >>> StringConverter.upgrade_mapper(dateparser, default=defaultdate)
        """
    if hasattr(func, '__call__'):
        cls._mapper.insert(-1, (cls._getsubdtype(default), func, default))
        return
    elif hasattr(func, '__iter__'):
        if isinstance(func[0], (tuple, list)):
            for _ in func:
                cls._mapper.insert(-1, _)
            return
        if default is None:
            default = [None] * len(func)
        else:
            default = list(default)
            default.append([None] * (len(func) - len(default)))
        for fct, dft in zip(func, default):
            cls._mapper.insert(-1, (cls._getsubdtype(dft), fct, dft))