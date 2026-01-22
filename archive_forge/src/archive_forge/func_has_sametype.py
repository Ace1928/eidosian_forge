import os
import abc
import numbers
import numpy as np
from . import polyutils as pu
def has_sametype(self, other):
    """Check if types match.

        .. versionadded:: 1.7.0

        Parameters
        ----------
        other : object
            Class instance.

        Returns
        -------
        bool : boolean
            True if other is same class as self

        """
    return isinstance(other, self.__class__)