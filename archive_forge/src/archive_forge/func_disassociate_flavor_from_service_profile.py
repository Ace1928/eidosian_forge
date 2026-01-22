from openstack import resource
from openstack import utils
def disassociate_flavor_from_service_profile(self, session, service_profile_id=None):
    flavor_id = self.id
    url = utils.urljoin(self.base_path, flavor_id, 'service_profiles', service_profile_id)
    session.delete(url)
    return None