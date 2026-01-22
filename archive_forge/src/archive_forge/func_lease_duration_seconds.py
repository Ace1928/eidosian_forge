from pprint import pformat
from six import iteritems
import re
@lease_duration_seconds.setter
def lease_duration_seconds(self, lease_duration_seconds):
    """
        Sets the lease_duration_seconds of this V1LeaseSpec.
        leaseDurationSeconds is a duration that candidates for a lease need to
        wait to force acquire it. This is measure against time of last observed
        RenewTime.

        :param lease_duration_seconds: The lease_duration_seconds of this
        V1LeaseSpec.
        :type: int
        """
    self._lease_duration_seconds = lease_duration_seconds