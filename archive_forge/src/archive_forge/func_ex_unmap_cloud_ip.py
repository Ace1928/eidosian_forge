import base64
from libcloud.utils.py3 import b, httplib
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.compute.types import Provider, NodeState
from libcloud.common.brightbox import BrightboxConnection
def ex_unmap_cloud_ip(self, cloud_ip_id):
    """
        Unmaps a cloud IP address from its current destination making
        it available to remap. This remains in the account's pool
        of addresses

        @note: This is an API extension for use on Brightbox

        :param  cloud_ip_id: The id of the cloud ip.
        :type   cloud_ip_id: ``str``

        :return: True if the unmap was successful.
        :rtype: ``bool``
        """
    response = self._post('/{}/cloud_ips/{}/unmap'.format(self.api_version, cloud_ip_id))
    return response.status == httplib.ACCEPTED