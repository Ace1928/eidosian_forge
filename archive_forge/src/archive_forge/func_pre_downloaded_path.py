import collections
from pathlib import Path
import string
from urllib.request import urlopen
import warnings
from cartopy import config
def pre_downloaded_path(self, format_dict):
    """
        The path on disk of the file that this resource represents, if it does
        not exist, then no further action will be taken with this path, and all
        further processing will be done using :meth:`target_path` instead.

        Parameters
        ----------
        format_dict
            The dictionary which is used to replace certain
            template variables. Subclasses should document which keys are
            expected as a minimum in their ``FORMAT_KEYS`` class attribute.

        """
    p = self._formatter.format(self.pre_downloaded_path_template, **format_dict)
    return None if p == '' else Path(p)