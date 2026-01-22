from pprint import pformat
from six import iteritems
import re
@boot_id.setter
def boot_id(self, boot_id):
    """
        Sets the boot_id of this V1NodeSystemInfo.
        Boot ID reported by the node.

        :param boot_id: The boot_id of this V1NodeSystemInfo.
        :type: str
        """
    if boot_id is None:
        raise ValueError('Invalid value for `boot_id`, must not be `None`')
    self._boot_id = boot_id