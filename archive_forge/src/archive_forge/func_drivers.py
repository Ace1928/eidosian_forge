from pprint import pformat
from six import iteritems
import re
@drivers.setter
def drivers(self, drivers):
    """
        Sets the drivers of this V1beta1CSINodeSpec.
        drivers is a list of information of all CSI Drivers existing on a node.
        If all drivers in the list are uninstalled, this can become empty.

        :param drivers: The drivers of this V1beta1CSINodeSpec.
        :type: list[V1beta1CSINodeDriver]
        """
    if drivers is None:
        raise ValueError('Invalid value for `drivers`, must not be `None`')
    self._drivers = drivers