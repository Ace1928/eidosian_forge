from suds import *
from suds.bindings.document import Document
from suds.bindings.rpc import RPC, Encoded
from suds.reader import DocumentReader
from suds.sax.element import Element
from suds.sudsobject import Object, Facade, Metadata
from suds.xsd import qualify, Namespace
from suds.xsd.query import ElementQuery
from suds.xsd.schema import Schema, SchemaCollection
import re
from . import soaparray
from urllib.parse import urljoin
from logging import getLogger
def setlocation(self, url, names=None):
    """
        Override the invocation location (URL) for service method.

        @param url: A URL location.
        @type url: A URL.
        @param names:  A list of method names. None=ALL
        @type names: [str,..]

        """
    for p in self.ports:
        for m in list(p.methods.values()):
            if names is None or m.name in names:
                m.location = url