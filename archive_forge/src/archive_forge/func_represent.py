from .error import *
from .nodes import *
import datetime, copyreg, types, base64, collections
def represent(self, data):
    node = self.represent_data(data)
    self.serialize(node)
    self.represented_objects = {}
    self.object_keeper = []
    self.alias_key = None