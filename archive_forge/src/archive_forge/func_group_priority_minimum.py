from pprint import pformat
from six import iteritems
import re
@group_priority_minimum.setter
def group_priority_minimum(self, group_priority_minimum):
    """
        Sets the group_priority_minimum of this V1APIServiceSpec.
        GroupPriorityMininum is the priority this group should have at least.
        Higher priority means that the group is preferred by clients over lower
        priority ones. Note that other versions of this group might specify even
        higher GroupPriorityMininum values such that the whole group gets a
        higher priority. The primary sort is based on GroupPriorityMinimum,
        ordered highest number to lowest (20 before 10). The secondary sort is
        based on the alphabetical comparison of the name of the object.  (v1.bar
        before v1.foo) We'd recommend something like: *.k8s.io (except
        extensions) at 18000 and PaaSes (OpenShift, Deis) are recommended to be
        in the 2000s

        :param group_priority_minimum: The group_priority_minimum of this
        V1APIServiceSpec.
        :type: int
        """
    if group_priority_minimum is None:
        raise ValueError('Invalid value for `group_priority_minimum`, must not be `None`')
    self._group_priority_minimum = group_priority_minimum