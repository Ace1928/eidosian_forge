from pprint import pformat
from six import iteritems
import re
@update_revision.setter
def update_revision(self, update_revision):
    """
        Sets the update_revision of this V1beta1StatefulSetStatus.
        updateRevision, if not empty, indicates the version of the StatefulSet
        used to generate Pods in the sequence
        [replicas-updatedReplicas,replicas)

        :param update_revision: The update_revision of this
        V1beta1StatefulSetStatus.
        :type: str
        """
    self._update_revision = update_revision