from pprint import pformat
from six import iteritems
import re
@condition_type.setter
def condition_type(self, condition_type):
    """
        Sets the condition_type of this V1PodReadinessGate.
        ConditionType refers to a condition in the pod's condition list with
        matching type.

        :param condition_type: The condition_type of this V1PodReadinessGate.
        :type: str
        """
    if condition_type is None:
        raise ValueError('Invalid value for `condition_type`, must not be `None`')
    self._condition_type = condition_type