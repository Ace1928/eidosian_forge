import logging
from typing import Any
from typing import Optional
from typing import Union
from xml.etree import ElementTree
import defusedxml.ElementTree
from saml2.validate import valid_instance
from saml2.version import version as __version__
def get_xml_string_with_self_contained_assertion_within_advice_encrypted_assertion(self, assertion_tag, advice_tag):
    for tmp_encrypted_assertion in self.assertion.advice.encrypted_assertion:
        if tmp_encrypted_assertion.encrypted_data is None:
            prefix_map = self.get_prefix_map([tmp_encrypted_assertion._to_element_tree().find(assertion_tag)])
            tree = self._to_element_tree()
            encs = tree.find(assertion_tag).find(advice_tag).findall(tmp_encrypted_assertion._to_element_tree().tag)
            for enc in encs:
                assertion = enc.find(assertion_tag)
                if assertion is not None:
                    self.set_prefixes(assertion, prefix_map)
    return ElementTree.tostring(tree, encoding='UTF-8').decode('utf-8')