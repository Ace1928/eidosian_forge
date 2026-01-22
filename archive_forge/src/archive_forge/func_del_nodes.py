from openstack.clustering.v1 import _async_resource
from openstack.common import metadata
from openstack import resource
from openstack import utils
def del_nodes(self, session, nodes, **params):
    data = {'nodes': nodes}
    data.update(params)
    body = {'del_nodes': data}
    return self.action(session, body)