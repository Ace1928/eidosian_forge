from pprint import pformat
from six import iteritems
import re
@grace_period_seconds.setter
def grace_period_seconds(self, grace_period_seconds):
    """
        Sets the grace_period_seconds of this V1DeleteOptions.
        The duration in seconds before the object should be deleted. Value must
        be non-negative integer. The value zero indicates delete immediately. If
        this value is nil, the default grace period for the specified type will
        be used. Defaults to a per object value if not specified. zero means
        delete immediately.

        :param grace_period_seconds: The grace_period_seconds of this
        V1DeleteOptions.
        :type: int
        """
    self._grace_period_seconds = grace_period_seconds