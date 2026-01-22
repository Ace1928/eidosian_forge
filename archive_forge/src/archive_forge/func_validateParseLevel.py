import types
from boto.gs.user import User
from boto.exception import InvalidCorsError
from xml.sax import handler
def validateParseLevel(self, tag, level):
    """Verify parse level for a given tag."""
    if self.parse_level != level:
        raise InvalidCorsError('Invalid tag %s at parse level %d: ' % (tag, self.parse_level))