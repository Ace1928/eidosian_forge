from collections import namedtuple
from testtools.tags import TagContext
@property
def current_tags(self):
    return self._tags.get_current_tags()