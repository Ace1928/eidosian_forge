from .error import *
from .nodes import *
import datetime, copyreg, types, base64, collections
def represent_set(self, data):
    value = {}
    for key in data:
        value[key] = None
    return self.represent_mapping('tag:yaml.org,2002:set', value)