from abc import ABCMeta
import locale
from collections.abc import MutableSequence
from urllib.parse import urlparse
from typing import cast, Any, Callable, ClassVar, Dict, List, \
from ..helpers import upper_camel_case, is_ncname, ordinal
from ..exceptions import ElementPathError, ElementPathTypeError, \
from ..namespaces import NamespacesType, XSD_NAMESPACE, XML_NAMESPACE, \
from ..collations import UNICODE_COLLATION_BASE_URI, UNICODE_CODEPOINT_COLLATION
from ..datatypes import UntypedAtomic, AtomicValueType, QName
from ..xpath_tokens import NargsType, XPathToken, ProxyToken, XPathFunction, XPathConstructor
from ..xpath_context import XPathContext, XPathSchemaContext
from ..sequence_types import is_sequence_type, match_sequence_type
from ..schema_proxy import AbstractSchemaProxy
from ..xpath1 import XPath1Parser
def evaluate_external_function(self_: XPathFunction, context: Optional[XPathContext]=None) -> Any:
    args = []
    for k in range(len(self_)):
        arg = self_.get_argument(context, index=k)
        args.append(arg)
    if sequence_types:
        for k, (arg, st) in enumerate(zip(args, sequence_types), start=1):
            if not match_sequence_type(arg, st, self):
                msg_ = f'{ordinal(k)} argument does not match sequence type {st!r}'
                raise xpath_error('XPDY0050', msg_)
        result = callback(*args)
        if not match_sequence_type(result, sequence_types[-1], self):
            msg_ = f'Result does not match sequence type {sequence_types[-1]!r}'
            raise xpath_error('XPDY0050', msg_)
        return result
    return callback(*args)