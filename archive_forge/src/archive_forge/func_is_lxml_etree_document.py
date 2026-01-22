import sys
import re
import io
import importlib
from typing import cast, Any, Counter, Iterator, Optional, MutableMapping, \
from .protocols import ElementProtocol, DocumentProtocol
import xml.etree.ElementTree as ElementTree
import xml.etree.ElementTree as PyElementTree  # noqa
import xml.etree  # noqa
def is_lxml_etree_document(obj: Any) -> bool:
    return is_etree_document(obj) and hasattr(obj, 'xpath') and hasattr(obj, 'xslt')