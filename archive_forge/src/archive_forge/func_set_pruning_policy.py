import collections
import threading
from typing import Callable, Deque, Optional, Set, Union
import dns.exception
import dns.immutable
import dns.name
import dns.node
import dns.rdataclass
import dns.rdataset
import dns.rdatatype
import dns.rdtypes.ANY.SOA
import dns.zone
def set_pruning_policy(self, policy: Optional[Callable[['Zone', Version], Optional[bool]]]) -> None:
    """Set the pruning policy for the zone.

        The *policy* function takes a `Version` and returns `True` if
        the version should be pruned, and `False` otherwise.  `None`
        may also be specified for policy, in which case the default policy
        is used.

        Pruning checking proceeds from the least version and the first
        time the function returns `False`, the checking stops.  I.e. the
        retained versions are always a consecutive sequence.
        """
    if policy is None:
        policy = self._default_pruning_policy
    with self._version_lock:
        self._pruning_policy = policy
        self._prune_versions_unlocked()