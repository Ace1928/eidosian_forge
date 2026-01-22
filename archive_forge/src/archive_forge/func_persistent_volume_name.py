from pprint import pformat
from six import iteritems
import re
@persistent_volume_name.setter
def persistent_volume_name(self, persistent_volume_name):
    """
        Sets the persistent_volume_name of this V1beta1VolumeAttachmentSource.
        Name of the persistent volume to attach.

        :param persistent_volume_name: The persistent_volume_name of this
        V1beta1VolumeAttachmentSource.
        :type: str
        """
    self._persistent_volume_name = persistent_volume_name