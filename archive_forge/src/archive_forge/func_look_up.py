import json
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import urlopen
from breezy.errors import BzrError
from breezy.trace import note
from breezy.urlutils import InvalidURL
def look_up(self, name, url, purpose=None):
    """See DirectoryService.look_up"""
    try:
        with urlopen('https://pypi.org/pypi/%s/json' % name) as f:
            data = json.load(f)
    except HTTPError as e:
        if e.status == 404:
            raise NoSuchPypiProject(name, url=url)
        raise
    url = find_repo_url(data)
    if url is None:
        raise PypiProjectWithoutRepositoryURL(name, url=url)
    return url