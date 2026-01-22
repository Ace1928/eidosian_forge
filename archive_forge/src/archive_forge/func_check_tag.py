from openstack import exceptions
from openstack import resource
from openstack import utils
def check_tag(self, session, tag):
    """Checks if tag exists on the entity.

        If the tag does not exist a 404 will be returned

        :param session: The session to use for making this request.
        :param tag: The tag as a string.
        """
    url = utils.urljoin(self.base_path, self.id, 'tags', tag)
    session = self._get_session(session)
    response = session.get(url)
    exceptions.raise_from_response(response, error_message='Tag does not exist')
    return self