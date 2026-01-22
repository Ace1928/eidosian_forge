from pprint import pformat
from six import iteritems
import re
@collision_count.setter
def collision_count(self, collision_count):
    """
        Sets the collision_count of this V1beta1StatefulSetStatus.
        collisionCount is the count of hash collisions for the StatefulSet. The
        StatefulSet controller uses this field as a collision avoidance
        mechanism when it needs to create the name for the newest
        ControllerRevision.

        :param collision_count: The collision_count of this
        V1beta1StatefulSetStatus.
        :type: int
        """
    self._collision_count = collision_count