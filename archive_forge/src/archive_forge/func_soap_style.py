import os
from ..exceptions import XMLSchemaException, XMLSchemaValueError
from ..names import XSD_NAMESPACE, WSDL_NAMESPACE, SOAP_NAMESPACE, \
from ..helpers import get_qname, local_name, get_extended_qname, get_prefixed_qname
from ..namespaces import NamespaceResourcesMap
from ..resources import fetch_resource
from ..documents import XmlDocument
from ..validators import XMLSchemaBase, XMLSchema10
@property
def soap_style(self):
    """The SOAP binding's style if any, `None` otherwise."""
    if self.soap_binding is not None:
        style = self.soap_binding.get('style')
        return style if style in ('rpc', 'document') else 'document'