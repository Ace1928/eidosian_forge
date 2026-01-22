import json
from decimal import Decimal, ROUND_UP
from types import ModuleType
from typing import cast, Any, Dict, Iterator, Iterable, Optional, Set, Union, Tuple
from xml.etree import ElementTree
from .exceptions import ElementPathError, xpath_error
from .namespaces import XSLT_XQUERY_SERIALIZATION_NAMESPACE
from .datatypes import AnyAtomicType, AnyURI, AbstractDateTime, \
from .xpath_nodes import XPathNode, ElementNode, AttributeNode, DocumentNode, \
from .xpath_tokens import XPathToken, XPathMap, XPathArray
from .protocols import EtreeElementProtocol, LxmlElementProtocol
def serialize_to_xml(elements: Iterable[Any], etree_module: Optional[ModuleType]=None, token: Optional['XPathToken']=None, **params: Any) -> str:
    if etree_module is None:
        etree_module = ElementTree
    item_separator = params.get('item_separator')
    character_map = params.get('character_map')
    cdata_section: Union[Set[str], Tuple[()]]
    kwargs = {}
    if 'xml_declaration' in params:
        kwargs['xml_declaration'] = params['xml_declaration']
    if 'standalone' in params:
        kwargs['standalone'] = params['standalone']
    if 'cdata_section' in params:
        cdata_section = {x.expanded_name for x in params['cdata_section']}
    else:
        cdata_section = ()
    method = kwargs.get('method', 'xml')
    if method == 'xhtml':
        method = 'html'
    chunks = []
    for item in iter_normalized(elements, item_separator):
        if isinstance(item, ElementNode):
            item = item.elem
        elif isinstance(item, (AttributeNode, NamespaceNode)):
            raise xpath_error('SENR0001', token=token)
        elif isinstance(item, TextNode):
            if item.parent is not None and item.parent.name in cdata_section:
                chunks.append(f'<![CDATA[{item.value}]]>')
            else:
                chunks.append(item.value)
            continue
        elif not isinstance(item, str):
            raise xpath_error('SENR0001', token=token)
        else:
            chunks.append(item)
            continue
        try:
            cks = etree_module.tostringlist(item, encoding='utf-8', method=method, **kwargs)
        except TypeError:
            ck = etree_module.tostring(item, encoding='utf-8', method=method)
            chunks.append(ck.decode('utf-8').rstrip(item.tail))
        else:
            if cks and cks[0].startswith(b'<?'):
                cks[0] = cks[0].replace(b"'", b'"')
            chunks.append(b'\n'.join(cks).decode('utf-8').rstrip(item.tail))
    if not character_map:
        return (item_separator or '').join(chunks)
    result = (item_separator or '').join(chunks)
    for character, map_string in character_map.items():
        result = result.replace(character, map_string)
    return result