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
def mktns(self):
    """
        Make the schema's target namespace.

        @return: namespace representation of the schema's targetNamespace
            value.
        @rtype: (prefix, URI)

        """
    tns = self.root.get('targetNamespace')
    tns_prefix = None
    if tns is not None:
        tns_prefix = self.root.findPrefix(tns)
    return (tns_prefix, tns)