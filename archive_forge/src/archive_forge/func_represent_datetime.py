from .error import *
from .nodes import *
import datetime, copyreg, types, base64, collections
def represent_datetime(self, data):
    value = data.isoformat(' ')
    return self.represent_scalar('tag:yaml.org,2002:timestamp', value)