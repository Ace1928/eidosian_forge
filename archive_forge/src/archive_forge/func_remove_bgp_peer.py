from openstack import exceptions
from openstack import resource
from openstack import utils
def remove_bgp_peer(self, session, peer_id):
    """Remove BGP Peer from a BGP Speaker

        :param session: The session to communicate through.
        :type session: :class:`~keystoneauth1.adapter.Adapter`
        :param peer_id: The ID of the peer to disassociate from the speaker.

        :raises: :class:`~openstack.exceptions.SDKException` on error.
        """
    url = utils.urljoin(self.base_path, self.id, 'remove_bgp_peer')
    body = {'bgp_peer_id': peer_id}
    self._put(session, url, body)