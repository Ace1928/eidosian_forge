from pprint import pformat
from six import iteritems
import re
@last_known_good.setter
def last_known_good(self, last_known_good):
    """
        Sets the last_known_good of this V1NodeConfigStatus.
        LastKnownGood reports the checkpointed config the node will fall back to
        when it encounters an error attempting to use the Assigned config. The
        Assigned config becomes the LastKnownGood config when the node
        determines that the Assigned config is stable and correct. This is
        currently implemented as a 10-minute soak period starting when the local
        record of Assigned config is updated. If the Assigned config is Active
        at the end of this period, it becomes the LastKnownGood. Note that if
        Spec.ConfigSource is reset to nil (use local defaults), the
        LastKnownGood is also immediately reset to nil, because the local
        default config is always assumed good. You should not make assumptions
        about the node's method of determining config stability and correctness,
        as this may change or become configurable in the future.

        :param last_known_good: The last_known_good of this V1NodeConfigStatus.
        :type: V1NodeConfigSource
        """
    self._last_known_good = last_known_good