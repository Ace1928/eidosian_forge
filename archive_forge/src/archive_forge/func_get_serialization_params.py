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
def get_serialization_params(params: Union[None, ElementNode, XPathMap]=None, token: Optional[XPathToken]=None) -> Dict['str', Any]:
    kwargs: Dict[str, Any] = {}
    character_map: Dict[str, str]
    if isinstance(params, XPathMap):
        if len(params[:]) > len(params.keys()):
            raise xpath_error('SEPM0019', token=token)
        for key, value in params.items():
            if not isinstance(key, str) or value is None:
                continue
            elif isinstance(value, UntypedAtomic):
                value = str(value)
                if value == 'true':
                    value = True
                elif value == 'false':
                    value = False
            if key == 'omit-xml-declaration':
                if not isinstance(value, bool):
                    raise xpath_error('XPTY0004', token=token)
                kwargs['xml_declaration'] = not value
            elif key == 'cdata-section-elements':
                if isinstance(value, XPathArray):
                    value = value.items()
                if not isinstance(value, list) or not all((isinstance(x, QName) for x in value)):
                    raise xpath_error('XPTY0004', token=token)
                kwargs['cdata_section'] = value
            elif key == 'method':
                if value not in ('html', 'xml', 'xhtml', 'text', 'adaptive', 'json'):
                    raise xpath_error('SEPM0017', token=token)
                kwargs[key] = value if value != 'xhtml' else 'html'
            elif key == 'indent':
                if not isinstance(value, bool):
                    raise xpath_error('XPTY0004', token=token)
                kwargs[key] = value
            elif key == 'item-separator':
                if not isinstance(value, str):
                    raise xpath_error('XPTY0004', token=token)
                kwargs['item_separator'] = value
            elif key == 'use-character-maps':
                if not isinstance(value, XPathMap):
                    raise xpath_error('XPTY0004', token=token)
                kwargs['character_map'] = character_map = {}
                for k, v in value.items():
                    if not isinstance(k, str) or not isinstance(v, str):
                        raise xpath_error('XPTY0004', token=token)
                    elif len(k) != 1:
                        msg = f'invalid character {k!r} in character map'
                        raise xpath_error('SEPM0016', msg, token)
                    else:
                        character_map[k] = v
            elif key == 'suppress-indentation':
                if isinstance(value, QName) or (isinstance(value, list) and all((isinstance(x, QName) for x in value))):
                    kwargs[key] = value
                else:
                    raise xpath_error('XPTY0004', token=token)
            elif key == 'standalone':
                if not value and isinstance(value, list):
                    pass
                elif isinstance(value, bool):
                    kwargs['standalone'] = value
                else:
                    if value not in ('yes', 'no', 'omit'):
                        raise xpath_error('XPTY0004', token=token)
                    if value != 'omit':
                        kwargs['standalone'] = value == 'yes'
            elif key == 'json-node-output-method':
                if not isinstance(value, (str, QName)):
                    raise xpath_error('XPTY0004', token=token)
                kwargs[key] = value
            elif key == 'allow-duplicate-names':
                if value is not None and (not isinstance(value, bool)):
                    raise xpath_error('XPTY0004', token=token)
                kwargs['allow_duplicate_names'] = value
            elif key == 'encoding':
                if not isinstance(value, str):
                    raise xpath_error('XPTY0004', token=token)
                kwargs[key] = value
            elif key == 'html-version':
                if not isinstance(value, (int, Decimal)):
                    raise xpath_error('XPTY0004', token=token)
                kwargs[key] = value
    elif isinstance(params, ElementNode):
        root = cast(Union[EtreeElementProtocol, LxmlElementProtocol], params.elem)
        if root.tag != SERIALIZATION_PARAMS:
            msg = 'output:serialization-parameters tag expected'
            raise xpath_error('XPTY0004', msg, token)
        if len(root) > len({e.tag for e in root}):
            raise xpath_error('SEPM0019', token=token)
        for child in root:
            if child.tag == SER_PARAM_OMIT_XML_DECLARATION:
                value = child.get('value')
                if value not in ('yes', 'no') or len(child.attrib) > 1:
                    raise xpath_error('SEPM0017', token=token)
                elif value == 'no':
                    kwargs['xml_declaration'] = True
            elif child.tag == SER_PARAM_USE_CHARACTER_MAPS:
                if len(child.attrib):
                    raise xpath_error('SEPM0017', token=token)
                kwargs['character_map'] = character_map = {}
                for e in child:
                    if e.tag != SER_PARAM_CHARACTER_MAP:
                        raise xpath_error('SEPM0017', token=token)
                    try:
                        character = e.attrib['character']
                        if character in character_map:
                            msg = 'duplicate character {!r} in character map'
                            raise xpath_error('SEPM0018', msg.format(character), token)
                        elif len(character) != 1:
                            msg = 'invalid character {!r} in character map'
                            raise xpath_error('SEPM0017', msg.format(character), token)
                        character_map[character] = e.attrib['map-string']
                    except KeyError as key:
                        msg = 'missing {} in character map'
                        raise xpath_error('SEPM0017', msg.format(key)) from None
                    else:
                        if len(e.attrib) > 2:
                            msg = 'invalid attribute in character map'
                            raise xpath_error('SEPM0017', msg)
            elif child.tag == SER_PARAM_METHOD:
                value = child.get('value')
                if value not in ('html', 'xml', 'xhtml', 'text') or len(child.attrib) > 1:
                    raise xpath_error('SEPM0017', token=token)
                kwargs['method'] = value if value != 'xhtml' else 'html'
            elif child.tag == SER_PARAM_INDENT:
                value = child.attrib.get('value', '')
                assert isinstance(value, str)
                value = value.strip()
                if value not in ('yes', 'no') or len(child.attrib) > 1:
                    raise xpath_error('SEPM0017', token=token)
            elif child.tag == SER_PARAM_ITEM_SEPARATOR:
                try:
                    kwargs['item_separator'] = child.attrib['value']
                except KeyError:
                    raise xpath_error('SEPM0017', token=token) from None
            elif child.tag == SER_PARAM_CDATA:
                pass
            elif child.tag == SER_PARAM_NO_INDENT:
                pass
            elif child.tag == SER_PARAM_STANDALONE:
                value = child.attrib.get('value', '')
                assert isinstance(value, str)
                value = value.strip()
                if value not in ('yes', 'no', 'omit') or len(child.attrib) > 1:
                    raise xpath_error('SEPM0017', token=token)
                if value != 'omit':
                    kwargs['standalone'] = value == 'yes'
            elif child.tag.startswith(f'{{{XSLT_XQUERY_SERIALIZATION_NAMESPACE}'):
                raise xpath_error('SEPM0017', token=token)
            elif not child.tag.startswith('{'):
                raise xpath_error('SEPM0017', token=token)
    return kwargs