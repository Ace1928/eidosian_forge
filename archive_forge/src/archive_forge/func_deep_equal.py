import math
from decimal import Decimal
from functools import cmp_to_key
from itertools import zip_longest
from typing import Any, Callable, Optional, Iterable, Iterator
from .protocols import ElementProtocol
from .exceptions import xpath_error
from .datatypes import UntypedAtomic, AnyURI, AbstractQName
from .collations import UNICODE_CODEPOINT_COLLATION, CollationManager
from .xpath_nodes import XPathNode, ElementNode, AttributeNode, NamespaceNode, \
from .xpath_tokens import XPathToken, XPathFunction, XPathMap, XPathArray
def deep_equal(seq1: Iterable[Any], seq2: Iterable[Any], collation: Optional[str]=None, token: Optional[XPathToken]=None) -> bool:
    etree_node_types = (ElementNode, CommentNode, ProcessingInstructionNode)

    def etree_deep_equal(e1: ElementProtocol, e2: ElementProtocol) -> bool:
        if cm.ne(e1.tag, e2.tag):
            return False
        elif cm.ne((e1.text or '').strip(), (e2.text or '').strip()):
            return False
        elif cm.ne((e1.tail or '').strip(), (e2.tail or '').strip()):
            return False
        elif len(e1) != len(e2) or len(e1.attrib) != len(e2.attrib):
            return False
        try:
            items1 = {(cm.strxfrm(k or ''), cm.strxfrm(v)) for k, v in e1.attrib.items()}
            items2 = {(cm.strxfrm(k or ''), cm.strxfrm(v)) for k, v in e2.attrib.items()}
        except TypeError:
            return False
        if items1 != items2:
            return False
        return all((etree_deep_equal(c1, c2) for c1, c2 in zip(e1, e2)))
    if collation is None:
        collation = UNICODE_CODEPOINT_COLLATION
    with CollationManager(collation, token=token) as cm:
        for value1, value2 in zip_longest(seq1, seq2):
            if isinstance(value1, XPathFunction) and (not isinstance(value1, (XPathMap, XPathArray))):
                raise xpath_error('FOTY0015', token=token)
            if isinstance(value2, XPathFunction) and (not isinstance(value2, (XPathMap, XPathArray))):
                raise xpath_error('FOTY0015', token=token)
            if (value1 is None) ^ (value2 is None):
                return False
            elif value1 is None:
                return True
            elif isinstance(value1, XPathNode) ^ isinstance(value2, XPathNode):
                return False
            elif isinstance(value1, XPathNode):
                assert isinstance(value2, XPathNode)
                if value1.kind != value2.kind:
                    return False
                elif isinstance(value1, etree_node_types):
                    assert isinstance(value2, etree_node_types)
                    if not etree_deep_equal(value1.elem, value2.elem):
                        return False
                elif isinstance(value1, DocumentNode):
                    assert isinstance(value2, DocumentNode)
                    for child1, child2 in zip_longest(value1, value2):
                        if child1 is None or child2 is None:
                            return False
                        elif child1.kind != child2.kind:
                            return False
                        elif isinstance(child1, etree_node_types):
                            assert isinstance(child2, etree_node_types)
                            if not etree_deep_equal(child1.elem, child2.elem):
                                return False
                        elif isinstance(child1, TextNode):
                            assert isinstance(child2, TextNode)
                            if cm.ne(child1.value, child2.value):
                                return False
                elif cm.ne(value1.value, value2.value):
                    return False
                elif isinstance(value1, AttributeNode):
                    if cm.ne(value1.name, value2.name):
                        return False
                elif isinstance(value1, NamespaceNode):
                    assert isinstance(value2, NamespaceNode)
                    if cm.ne(value1.prefix, value2.prefix):
                        return False
            else:
                try:
                    if isinstance(value1, bool):
                        if not isinstance(value2, bool) or value1 is not value2:
                            return False
                    elif isinstance(value2, bool):
                        return False
                    if isinstance(value1, AbstractQName):
                        if not isinstance(value2, AbstractQName) or value1 != value2:
                            return False
                    elif isinstance(value2, AbstractQName):
                        return False
                    elif isinstance(value1, (str, AnyURI, UntypedAtomic)) and isinstance(value2, (str, AnyURI, UntypedAtomic)):
                        if cm.strcoll(str(value1), str(value2)):
                            return False
                    elif isinstance(value1, UntypedAtomic) or isinstance(value2, UntypedAtomic):
                        return False
                    elif isinstance(value1, float):
                        if math.isnan(value1):
                            if not math.isnan(value2):
                                return False
                        elif math.isinf(value1):
                            if value1 != value2:
                                return False
                        elif isinstance(value2, Decimal):
                            if value1 != float(value2):
                                return False
                        elif not isinstance(value2, (value1.__class__, int)):
                            return False
                        elif value1 != value2:
                            return False
                    elif isinstance(value2, float):
                        if math.isnan(value2):
                            return False
                        elif math.isinf(value2):
                            if value1 != value2:
                                return False
                        elif isinstance(value1, Decimal):
                            if value2 != float(value1):
                                return False
                        elif not isinstance(value1, (value2.__class__, int)):
                            return False
                        elif value1 != value2:
                            return False
                    elif value1 != value2:
                        return False
                except TypeError:
                    return False
    return True