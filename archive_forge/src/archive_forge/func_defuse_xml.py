import sys
import re
import io
import importlib
from typing import cast, Any, Counter, Iterator, Optional, MutableMapping, \
from .protocols import ElementProtocol, DocumentProtocol
import xml.etree.ElementTree as ElementTree
import xml.etree.ElementTree as PyElementTree  # noqa
import xml.etree  # noqa
def defuse_xml(xml_source: Union[str, bytes]) -> Union[str, bytes]:
    resource: Any
    if isinstance(xml_source, str):
        resource = io.StringIO(xml_source)
    else:
        resource = io.BytesIO(xml_source)
    safe_parser = SafeXMLParser(target=PyElementTree.TreeBuilder())
    try:
        for _ in PyElementTree.iterparse(resource, ('start',), safe_parser):
            break
    except PyElementTree.ParseError as err:
        msg = str(err)
        if 'Entities are forbidden' in msg or 'Unparsed entities are forbidden' in msg or 'External references are forbidden' in msg:
            raise
    return xml_source