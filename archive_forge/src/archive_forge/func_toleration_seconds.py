from pprint import pformat
from six import iteritems
import re
@toleration_seconds.setter
def toleration_seconds(self, toleration_seconds):
    """
        Sets the toleration_seconds of this V1Toleration.
        TolerationSeconds represents the period of time the toleration (which
        must be of effect NoExecute, otherwise this field is ignored) tolerates
        the taint. By default, it is not set, which means tolerate the taint
        forever (do not evict). Zero and negative values will be treated as 0
        (evict immediately) by the system.

        :param toleration_seconds: The toleration_seconds of this V1Toleration.
        :type: int
        """
    self._toleration_seconds = toleration_seconds