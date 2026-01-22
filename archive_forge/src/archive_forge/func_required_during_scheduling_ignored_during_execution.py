from pprint import pformat
from six import iteritems
import re
@required_during_scheduling_ignored_during_execution.setter
def required_during_scheduling_ignored_during_execution(self, required_during_scheduling_ignored_during_execution):
    """
        Sets the required_during_scheduling_ignored_during_execution of this
        V1PodAffinity.
        If the affinity requirements specified by this field are not met at
        scheduling time, the pod will not be scheduled onto the node. If the
        affinity requirements specified by this field cease to be met at some
        point during pod execution (e.g. due to a pod label update), the system
        may or may not try to eventually evict the pod from its node. When there
        are multiple elements, the lists of nodes corresponding to each
        podAffinityTerm are intersected, i.e. all terms must be satisfied.

        :param required_during_scheduling_ignored_during_execution: The
        required_during_scheduling_ignored_during_execution of this
        V1PodAffinity.
        :type: list[V1PodAffinityTerm]
        """
    self._required_during_scheduling_ignored_during_execution = required_during_scheduling_ignored_during_execution