from openstack import exceptions
from openstack import resource
from openstack import utils
def get_metadata_item(self, session, key):
    """Get the single metadata item on the entity.

        If the metadata key does not exist a 404 will be returned

        :param session: The session to use for making this request.
        :param str key: The key of a metadata item.
        """
    url = utils.urljoin(self.base_path, self.id, 'metadata', key)
    response = session.get(url)
    exceptions.raise_from_response(response, error_message='Metadata item does not exist')
    meta = response.json().get('meta', {})
    metadata = self.metadata or {}
    metadata[key] = meta.get(key)
    self._body.attributes.update({'metadata': metadata})
    return self