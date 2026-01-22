import os
import warnings
from enchant.errors import Error, DictNotFoundError
from enchant.utils import get_default_language
from enchant.pypwl import PyPWL
def dict_exists(self, tag):
    """Check availability of a dictionary.

        This method checks whether there is a dictionary available for
        the language specified by 'tag'.  It returns True if a dictionary
        is available, and False otherwise.
        """
    self._check_this()
    val = _e.broker_dict_exists(self._this, tag.encode())
    return bool(val)