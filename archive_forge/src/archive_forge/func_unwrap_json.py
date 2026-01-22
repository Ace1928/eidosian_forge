import re
from ovs.db import error
def unwrap_json(json, name, types, desc):
    if not isinstance(json, (list, tuple)) or len(json) != 2 or json[0] != name or (not isinstance(json[1], tuple(types))):
        raise error.Error('expected ["%s", <%s>]' % (name, desc), json)
    return json[1]