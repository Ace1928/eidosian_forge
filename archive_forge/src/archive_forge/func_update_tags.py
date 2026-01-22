import urllib.parse
from keystoneclient import base
from keystoneclient import exceptions
from keystoneclient.i18n import _
def update_tags(self, project, tags):
    """Update tag list of a project.

        Replaces current tag list with list specified in tags parameter.

        :param project: project to update.
        :param tags: list of str tag names to add to the project

        :returns: list of tags

        """
    url = '/projects/%s/tags' % base.getid(project)
    for tag in tags:
        tag = urllib.parse.quote(tag)
    resp, body = self.client.put(url, body={'tags': tags})
    return self._prepare_return_value(resp, body['tags'])