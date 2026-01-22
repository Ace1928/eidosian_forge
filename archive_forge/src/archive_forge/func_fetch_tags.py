from openstack import exceptions
from openstack import resource
from openstack import utils
def fetch_tags(self, session):
    """Lists tags set on the entity.

        :param session: The session to use for making this request.
        :return: The list with tags attached to the entity
        """
    url = utils.urljoin(self.base_path, self.id, 'tags')
    session = self._get_session(session)
    response = session.get(url)
    exceptions.raise_from_response(response)
    json = response.json()
    if 'tags' in json:
        self._body.attributes.update({'tags': json['tags']})
    return self