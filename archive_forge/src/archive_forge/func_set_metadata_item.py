from openstack import exceptions
from openstack import resource
from openstack import utils
def set_metadata_item(self, session, key, value):
    """Create or replace single metadata item to the resource.

        :param session: The session to use for making this request.
        :param str key: The key for the metadata item.
        :param str value: The value.
        """
    url = utils.urljoin(self.base_path, self.id, 'metadata', key)
    response = session.put(url, json={'meta': {key: value}})
    exceptions.raise_from_response(response)
    metadata = self.metadata
    metadata[key] = value
    self._body.attributes.update({'metadata': metadata})
    return self