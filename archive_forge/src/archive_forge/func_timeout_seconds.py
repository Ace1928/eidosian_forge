from pprint import pformat
from six import iteritems
import re
@timeout_seconds.setter
def timeout_seconds(self, timeout_seconds):
    """
        Sets the timeout_seconds of this V1ClientIPConfig.
        timeoutSeconds specifies the seconds of ClientIP type session sticky
        time. The value must be >0 && <=86400(for 1 day) if ServiceAffinity ==
        "ClientIP". Default value is 10800(for 3 hours).

        :param timeout_seconds: The timeout_seconds of this V1ClientIPConfig.
        :type: int
        """
    self._timeout_seconds = timeout_seconds