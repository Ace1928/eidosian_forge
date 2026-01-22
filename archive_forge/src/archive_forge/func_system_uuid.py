from pprint import pformat
from six import iteritems
import re
@system_uuid.setter
def system_uuid(self, system_uuid):
    """
        Sets the system_uuid of this V1NodeSystemInfo.
        SystemUUID reported by the node. For unique machine identification
        MachineID is preferred. This field is specific to Red Hat hosts
        https://access.redhat.com/documentation/en-US/Red_Hat_Subscription_Management/1/html/RHSM/getting-system-uuid.html

        :param system_uuid: The system_uuid of this V1NodeSystemInfo.
        :type: str
        """
    if system_uuid is None:
        raise ValueError('Invalid value for `system_uuid`, must not be `None`')
    self._system_uuid = system_uuid