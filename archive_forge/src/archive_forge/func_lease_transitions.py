from pprint import pformat
from six import iteritems
import re
@lease_transitions.setter
def lease_transitions(self, lease_transitions):
    """
        Sets the lease_transitions of this V1LeaseSpec.
        leaseTransitions is the number of transitions of a lease between
        holders.

        :param lease_transitions: The lease_transitions of this V1LeaseSpec.
        :type: int
        """
    self._lease_transitions = lease_transitions