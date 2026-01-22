import sys
import re
import os
from configparser import RawConfigParser
def sections(self):
    """
        Return the section headers of the config file.

        Parameters
        ----------
        None

        Returns
        -------
        keys : list of str
            The list of section headers.

        """
    return list(self._sections.keys())