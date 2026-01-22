from pprint import pformat
from six import iteritems
import re
@propagation_policy.setter
def propagation_policy(self, propagation_policy):
    """
        Sets the propagation_policy of this V1DeleteOptions.
        Whether and how garbage collection will be performed. Either this field
        or OrphanDependents may be set, but not both. The default policy is
        decided by the existing finalizer set in the metadata.finalizers and the
        resource-specific default policy. Acceptable values are: 'Orphan' -
        orphan the dependents; 'Background' - allow the garbage collector to
        delete the dependents in the background; 'Foreground' - a cascading
        policy that deletes all dependents in the foreground.

        :param propagation_policy: The propagation_policy of this
        V1DeleteOptions.
        :type: str
        """
    self._propagation_policy = propagation_policy