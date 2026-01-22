from pprint import pformat
from six import iteritems
import re
@priority_class_name.setter
def priority_class_name(self, priority_class_name):
    """
        Sets the priority_class_name of this V1PodSpec.
        If specified, indicates the pod's priority. "system-node-critical" and
        "system-cluster-critical" are two special keywords which indicate the
        highest priorities with the former being the highest priority. Any other
        name must be defined by creating a PriorityClass object with that name.
        If not specified, the pod priority will be default or zero if there is
        no default.

        :param priority_class_name: The priority_class_name of this V1PodSpec.
        :type: str
        """
    self._priority_class_name = priority_class_name