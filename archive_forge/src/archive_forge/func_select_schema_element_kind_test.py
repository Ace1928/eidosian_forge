import math
import operator
from copy import copy
from decimal import Decimal, DivisionByZero
from ..exceptions import ElementPathError
from ..helpers import OCCURRENCE_INDICATORS, numeric_equal, numeric_not_equal, \
from ..namespaces import XSD_NAMESPACE, XSD_NOTATION, XSD_ANY_ATOMIC_TYPE, \
from ..datatypes import get_atomic_value, UntypedAtomic, QName, AnyURI, \
from ..xpath_nodes import ElementNode, DocumentNode, XPathNode, AttributeNode
from ..sequence_types import is_instance
from ..xpath_context import XPathSchemaContext
from ..xpath_tokens import XPathFunction
from .xpath2_parser import XPath2Parser
@method(function('schema-element', nargs=1, label='kind test'))
def select_schema_element_kind_test(self, context=None):
    if context is None:
        raise self.missing_context()
    element_name = self[0].source
    qname = get_expanded_name(element_name, self.parser.namespaces)
    if self.parser.schema is not None:
        for _ in context.iter_children_or_self():
            if self.parser.schema.get_element(qname) is None and self.parser.schema.get_substitution_group(qname) is None:
                raise self.error('XPST0008', 'element %r not found in schema' % element_name)
            if isinstance(context.item, ElementNode) and context.item.elem.tag == qname:
                yield context.item
                return
    if not isinstance(context, XPathSchemaContext):
        raise self.error('XPST0008', 'schema element %r not found' % element_name)