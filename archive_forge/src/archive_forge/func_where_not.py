import csv
import copy
from fnmatch import fnmatch
import json
from io import StringIO
def where_not(self, *args, **kwargs):
    """ Filters the object. Conditions must not be matched.

            Paramaters
            ----------
            args:
                Value must not be matched in the key or the value of an
                entry.
            kwargs:
                Value for a specific key must not be matched in an entry.

            Returns
            -------
            A :class:`JsonTable` containing the not matches.
        """
    return self.__class__(get_where_not(self.data, *args, **kwargs), self.order_by)