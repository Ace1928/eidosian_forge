from pprint import pformat
from six import iteritems
import re
@assigned.setter
def assigned(self, assigned):
    """
        Sets the assigned of this V1NodeConfigStatus.
        Assigned reports the checkpointed config the node will try to use. When
        Node.Spec.ConfigSource is updated, the node checkpoints the associated
        config payload to local disk, along with a record indicating intended
        config. The node refers to this record to choose its config checkpoint,
        and reports this record in Assigned. Assigned only updates in the status
        after the record has been checkpointed to disk. When the Kubelet is
        restarted, it tries to make the Assigned config the Active config by
        loading and validating the checkpointed payload identified by Assigned.

        :param assigned: The assigned of this V1NodeConfigStatus.
        :type: V1NodeConfigSource
        """
    self._assigned = assigned