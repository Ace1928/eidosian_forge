from .error import *
from .nodes import *
import datetime, copyreg, types, base64, collections
def represent_dict(self, data):
    return self.represent_mapping('tag:yaml.org,2002:map', data)