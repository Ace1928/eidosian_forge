from .error import *
from .nodes import *
import datetime, copyreg, types, base64, collections
def represent_binary(self, data):
    if hasattr(base64, 'encodebytes'):
        data = base64.encodebytes(data).decode('ascii')
    else:
        data = base64.encodestring(data).decode('ascii')
    return self.represent_scalar('tag:yaml.org,2002:binary', data, style='|')