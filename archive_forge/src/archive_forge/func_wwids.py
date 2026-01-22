from pprint import pformat
from six import iteritems
import re
@wwids.setter
def wwids(self, wwids):
    """
        Sets the wwids of this V1FCVolumeSource.
        Optional: FC volume world wide identifiers (wwids) Either wwids or
        combination of targetWWNs and lun must be set, but not both
        simultaneously.

        :param wwids: The wwids of this V1FCVolumeSource.
        :type: list[str]
        """
    self._wwids = wwids