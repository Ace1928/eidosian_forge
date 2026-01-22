from suds import *
from suds.xsd import *
from suds.xsd.depsort import dependency_sort
from suds.xsd.sxbuiltin import *
from suds.xsd.sxbase import SchemaObject
from suds.xsd.sxbasic import Factory as BasicFactory
from suds.xsd.sxbuiltin import Factory as BuiltinFactory
from suds.sax import splitPrefix, Namespace
from suds.sax.element import Element
from logging import getLogger
def autoblend(self):
    """
        Ensure that all schemas within the collection import each other which
        has a blending effect.

        @return: self
        @rtype: L{SchemaCollection}

        """
    namespaces = list(self.namespaces.keys())
    for s in self.children:
        for ns in namespaces:
            tns = s.root.get('targetNamespace')
            if tns == ns:
                continue
            for imp in s.root.getChildren('import'):
                if imp.get('namespace') == ns:
                    continue
            imp = Element('import', ns=Namespace.xsdns)
            imp.set('namespace', ns)
            s.root.append(imp)
    return self