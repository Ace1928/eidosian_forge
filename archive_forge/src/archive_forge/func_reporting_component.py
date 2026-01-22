from pprint import pformat
from six import iteritems
import re
@reporting_component.setter
def reporting_component(self, reporting_component):
    """
        Sets the reporting_component of this V1Event.
        Name of the controller that emitted this Event, e.g.
        `kubernetes.io/kubelet`.

        :param reporting_component: The reporting_component of this V1Event.
        :type: str
        """
    self._reporting_component = reporting_component