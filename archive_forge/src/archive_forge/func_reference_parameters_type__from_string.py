import saml2
from saml2 import SamlBase
def reference_parameters_type__from_string(xml_string):
    return saml2.create_class_from_xml_string(ReferenceParametersType_, xml_string)