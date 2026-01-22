from xml.sax.saxutils import XMLGenerator
from xml.sax.xmlreader import AttributesImpl
import platform
from inspect import isgenerator
def startNamespaceDecl(self, prefix, uri):
    self.namespace_declarations[prefix or ''] = uri