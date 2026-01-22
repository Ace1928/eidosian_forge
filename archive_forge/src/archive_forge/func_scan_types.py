from . import schema
from .jsonutil import get_column
from .search import Search
def scan_types(self):
    """ Returns the datatypes used at the scan level in this
            database.

            See Also
            --------
            :func:`Inspector.set_autolearn`
        """
    return self._resource_types('scan')