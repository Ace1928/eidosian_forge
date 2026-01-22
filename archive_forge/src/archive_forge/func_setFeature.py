import copy
import xml.dom
from xml.dom.NodeFilter import NodeFilter
def setFeature(self, name, state):
    if self.supportsFeature(name):
        state = state and 1 or 0
        try:
            settings = self._settings[_name_xform(name), state]
        except KeyError:
            raise xml.dom.NotSupportedErr('unsupported feature: %r' % (name,)) from None
        else:
            for name, value in settings:
                setattr(self._options, name, value)
    else:
        raise xml.dom.NotFoundErr('unknown feature: ' + repr(name))