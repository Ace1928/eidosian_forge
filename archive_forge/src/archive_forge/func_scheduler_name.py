from pprint import pformat
from six import iteritems
import re
@scheduler_name.setter
def scheduler_name(self, scheduler_name):
    """
        Sets the scheduler_name of this V1PodSpec.
        If specified, the pod will be dispatched by specified scheduler. If not
        specified, the pod will be dispatched by default scheduler.

        :param scheduler_name: The scheduler_name of this V1PodSpec.
        :type: str
        """
    self._scheduler_name = scheduler_name