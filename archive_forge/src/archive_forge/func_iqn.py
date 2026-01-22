from pprint import pformat
from six import iteritems
import re
@iqn.setter
def iqn(self, iqn):
    """
        Sets the iqn of this V1ISCSIVolumeSource.
        Target iSCSI Qualified Name.

        :param iqn: The iqn of this V1ISCSIVolumeSource.
        :type: str
        """
    if iqn is None:
        raise ValueError('Invalid value for `iqn`, must not be `None`')
    self._iqn = iqn