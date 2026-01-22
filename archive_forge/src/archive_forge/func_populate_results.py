import re
from w3lib.html import strip_html5_whitespace
from extruct.utils import parse_html
def populate_results(node, main_attrib):
    node_attrib = node.attrib
    if main_attrib not in node_attrib:
        return
    name = node.attrib[main_attrib]
    lower_name = get_lower_attrib(name)
    if lower_name in _DC_ELEMENTS:
        node.attrib.update({'URI': _DC_ELEMENTS[lower_name]})
        elements.append(attrib_to_dict(node.attrib))
    elif lower_name in _DC_TERMS:
        node.attrib.update({'URI': _DC_TERMS[lower_name]})
        terms.append(attrib_to_dict(node.attrib))