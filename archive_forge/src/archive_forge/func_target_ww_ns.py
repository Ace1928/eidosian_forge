from pprint import pformat
from six import iteritems
import re
@target_ww_ns.setter
def target_ww_ns(self, target_ww_ns):
    """
        Sets the target_ww_ns of this V1FCVolumeSource.
        Optional: FC target worldwide names (WWNs)

        :param target_ww_ns: The target_ww_ns of this V1FCVolumeSource.
        :type: list[str]
        """
    self._target_ww_ns = target_ww_ns