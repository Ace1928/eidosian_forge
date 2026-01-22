from pprint import pformat
from six import iteritems
import re
@qos_class.setter
def qos_class(self, qos_class):
    """
        Sets the qos_class of this V1PodStatus.
        The Quality of Service (QOS) classification assigned to the pod based on
        resource requirements See PodQOSClass type for available QOS classes
        More info:
        https://git.k8s.io/community/contributors/design-proposals/node/resource-qos.md

        :param qos_class: The qos_class of this V1PodStatus.
        :type: str
        """
    self._qos_class = qos_class