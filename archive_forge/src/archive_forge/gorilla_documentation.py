import collections
import copy
import inspect
import logging
import pkgutil
import sys
import types
Update some attributes.

        If a 'settings' attribute is passed as a dict, then it will update the
        content of the settings, if any, instead of completely overwriting it.

        Parameters
        ----------
        kwargs
            Attributes to update.

        Raises
        ------
        ValueError
            The setting doesn't exist.
        