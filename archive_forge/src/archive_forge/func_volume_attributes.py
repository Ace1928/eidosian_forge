from pprint import pformat
from six import iteritems
import re
@volume_attributes.setter
def volume_attributes(self, volume_attributes):
    """
        Sets the volume_attributes of this V1CSIPersistentVolumeSource.
        Attributes of the volume to publish.

        :param volume_attributes: The volume_attributes of this
        V1CSIPersistentVolumeSource.
        :type: dict(str, str)
        """
    self._volume_attributes = volume_attributes