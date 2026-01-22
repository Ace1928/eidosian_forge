from collections.abc import MutableMapping, MutableSequence
from typing import TYPE_CHECKING, Any, Optional, List, Dict, Type, Union, Tuple
from ..aliases import NamespacesType, BaseXsdType
from .default import ElementData, XMLSchemaConverter
def element_encode(self, obj: Any, xsd_element: 'XsdElement', level: int=0) -> ElementData:
    tag = xsd_element.qualified_name if level == 0 else xsd_element.name
    if not self.strip_namespaces:
        try:
            self.update(((k if k != '$' else '', v) for k, v in obj['@xmlns'].items()))
        except KeyError:
            pass
    try:
        element_data = obj[self.map_qname(xsd_element.name)]
    except KeyError:
        try:
            element_data = obj[xsd_element.name]
        except KeyError:
            element_data = obj
    text = None
    content: List[Tuple[Union[str, int], Any]] = []
    attributes = {}
    for name, value in element_data.items():
        if name == '@xmlns':
            continue
        elif name == '$':
            text = value
        elif name[0] == '$' and name[1:].isdigit():
            content.append((int(name[1:]), value))
        elif name[0] == '@':
            attr_name = name[1:]
            ns_name = self.unmap_qname(attr_name, xsd_element.attributes)
            attributes[ns_name] = value
        elif not isinstance(value, MutableSequence) or not value:
            content.append((self.unmap_qname(name), value))
        elif isinstance(value[0], (MutableMapping, MutableSequence)):
            ns_name = self.unmap_qname(name)
            for item in value:
                content.append((ns_name, item))
        else:
            xsd_group = xsd_element.type.model_group
            if xsd_group is None:
                xsd_group = xsd_element.any_type.model_group
                assert xsd_group is not None
            ns_name = self.unmap_qname(name)
            for xsd_child in xsd_group.iter_elements():
                matched_element = xsd_child.match(ns_name, resolve=True)
                if matched_element is not None:
                    if matched_element.type and matched_element.type.is_list():
                        content.append((ns_name, value))
                    else:
                        content.extend(((ns_name, item) for item in value))
                    break
            else:
                content.append((ns_name, value))
    return ElementData(tag, text, content, attributes)