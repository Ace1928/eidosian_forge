from openstack import exceptions
from openstack import resource
from openstack import utils
def remove_agent_from_network(self, session, network_id):
    body = {'network_id': network_id}
    url = utils.urljoin(self.base_path, self.id, 'dhcp-networks', network_id)
    session.delete(url, json=body)