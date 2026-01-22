from pprint import pformat
from six import iteritems
import re
@preconditions.setter
def preconditions(self, preconditions):
    """
        Sets the preconditions of this V1DeleteOptions.
        Must be fulfilled before a deletion is carried out. If not possible, a
        409 Conflict status will be returned.

        :param preconditions: The preconditions of this V1DeleteOptions.
        :type: V1Preconditions
        """
    self._preconditions = preconditions