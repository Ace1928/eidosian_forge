from suds import *
from suds.xsd import *
from suds.sax.element import Element
from suds.sax import Namespace
from logging import getLogger
def default_namespace(self):
    return self.root.defaultNamespace()