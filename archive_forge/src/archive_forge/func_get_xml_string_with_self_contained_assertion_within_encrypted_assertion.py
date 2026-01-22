import logging
from typing import Any
from typing import Optional
from typing import Union
from xml.etree import ElementTree
import defusedxml.ElementTree
from saml2.validate import valid_instance
from saml2.version import version as __version__
def get_xml_string_with_self_contained_assertion_within_encrypted_assertion(self, assertion_tag):
    """Makes a encrypted assertion only containing self contained
        namespaces.

        :param assertion_tag: Tag for the assertion to be transformed.
        :return: A new samlp.Resonse in string representation.
        """
    prefix_map = self.get_prefix_map([self.encrypted_assertion._to_element_tree().find(assertion_tag)])
    tree = self._to_element_tree()
    self.set_prefixes(tree.find(self.encrypted_assertion._to_element_tree().tag).find(assertion_tag), prefix_map)
    return ElementTree.tostring(tree, encoding='UTF-8').decode('utf-8')