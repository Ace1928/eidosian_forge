from openstack import exceptions
from openstack import resource
from openstack import utils
def fetch_metadata(self, session):
    """Lists metadata set on the entity.

        :param session: The session to use for making this request.
        :return: The dictionary with metadata attached to the entity
        """
    url = utils.urljoin(self.base_path, self.id, 'metadata')
    response = session.get(url)
    exceptions.raise_from_response(response)
    json = response.json()
    if 'metadata' in json:
        self._body.attributes.update({'metadata': json['metadata']})
    return self