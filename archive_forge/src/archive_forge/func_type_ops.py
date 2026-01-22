from openstack import resource
from openstack import utils
def type_ops(self, session):
    url = utils.urljoin(self.base_path, self.id, 'ops')
    resp = session.get(url)
    return resp.json()