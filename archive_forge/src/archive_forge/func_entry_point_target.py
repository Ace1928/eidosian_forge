import logging
import operator
from . import _cache
from .exception import NoMatches
@property
def entry_point_target(self):
    """The module and attribute referenced by this extension's entry_point.

        :return: A string representation of the target of the entry point in
            'dotted.module:object' format.
        """
    return self.entry_point.value