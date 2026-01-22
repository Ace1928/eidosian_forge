from .error import *
from .nodes import *
import datetime, copyreg, types, base64, collections
def represent_scalar(self, tag, value, style=None):
    if style is None:
        style = self.default_style
    node = ScalarNode(tag, value, style=style)
    if self.alias_key is not None:
        self.represented_objects[self.alias_key] = node
    return node