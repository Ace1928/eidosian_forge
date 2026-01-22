import collections
from pathlib import Path
import string
from urllib.request import urlopen
import warnings
from cartopy import config
def target_path(self, format_dict):
    """
        The path on disk of the file that this resource represents, must
        either exist, or be writable by the current user. This method
        does not check either of these conditions.

        Parameters
        ----------
        format_dict
            The dictionary which is used to replace certain
            template variables. Subclasses should document which keys are
            expected as a minimum in their ``FORMAT_KEYS`` class attribute.

        """
    return Path(self._formatter.format(self.target_path_template, **format_dict))