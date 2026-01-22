import sys
import re
import io
import importlib
from typing import cast, Any, Counter, Iterator, Optional, MutableMapping, \
from .protocols import ElementProtocol, DocumentProtocol
import xml.etree.ElementTree as ElementTree
import xml.etree.ElementTree as PyElementTree  # noqa
import xml.etree  # noqa
def external_entity_reference(self, context, base, system_id, public_id):
    raise PyElementTree.ParseError('External references are forbidden (system_id={!r}, public_id={!r})'.format(system_id, public_id))