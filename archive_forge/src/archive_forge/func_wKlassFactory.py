from reportlab.graphics import shapes
from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from weakref import ref as weakref_ref
def wKlassFactory(self, Klass):

    class WKlass(Klass, CloneMixin):

        def __getattr__(self, name):
            try:
                return self.__class__.__bases__[0].__getattr__(self, name)
            except:
                parent = self.parent
                c = parent._children
                x = self.__propholder_index__
                while x:
                    if x in c:
                        return getattr(c[x], name)
                    x = x[:-1]
                return getattr(parent, name)

        @property
        def parent(self):
            return self.__propholder_parent__()
    return WKlass