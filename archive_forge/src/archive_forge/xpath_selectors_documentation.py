from typing import TYPE_CHECKING, Any, Dict, Optional, Iterator, Union, Type
from .namespaces import NamespacesType
from .tree_builders import RootArgType
from .xpath_context import XPathContext
from .xpath2 import XPath2Parser

        Creates an XPath selector generator for apply the instance's XPath expression
        on *root* Element.

        :param root: the root of the XML document, usually an ElementTree instance         or an Element.
        :param kwargs: other optional parameters for the XPath dynamic context.
        :return: a generator of the XPath expression results.
        