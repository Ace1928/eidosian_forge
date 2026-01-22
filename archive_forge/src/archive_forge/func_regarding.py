from pprint import pformat
from six import iteritems
import re
@regarding.setter
def regarding(self, regarding):
    """
        Sets the regarding of this V1beta1Event.
        The object this Event is about. In most cases it's an Object reporting
        controller implements. E.g. ReplicaSetController implements ReplicaSets
        and this event is emitted because it acts on some changes in a
        ReplicaSet object.

        :param regarding: The regarding of this V1beta1Event.
        :type: V1ObjectReference
        """
    self._regarding = regarding