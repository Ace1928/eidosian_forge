from .error import *
from .nodes import *
import datetime, copyreg, types, base64, collections
def represent_module(self, data):
    return self.represent_scalar('tag:yaml.org,2002:python/module:' + data.__name__, '')