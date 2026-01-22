from pprint import pformat
from six import iteritems
import re
@preference.setter
def preference(self, preference):
    """
        Sets the preference of this V1PreferredSchedulingTerm.
        A node selector term, associated with the corresponding weight.

        :param preference: The preference of this V1PreferredSchedulingTerm.
        :type: V1NodeSelectorTerm
        """
    if preference is None:
        raise ValueError('Invalid value for `preference`, must not be `None`')
    self._preference = preference