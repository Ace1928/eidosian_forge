import re
import zlib
import base64
from types import MappingProxyType
from numba.core import utils
def make_prop(name, option):

    def getter(self):
        return self._values.get(name, option.default)

    def setter(self, val):
        self._values[name] = option.type(val)

    def delter(self):
        del self._values[name]
    return property(getter, setter, delter, option.doc)