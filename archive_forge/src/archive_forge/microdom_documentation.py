from __future__ import annotations
import re
import warnings
from io import BytesIO, StringIO
from incremental import Version, getVersionString
from twisted.python.compat import ioType
from twisted.python.util import InsensitiveDict
from twisted.web.sux import ParseError, XMLParser

        Serialize this L{Element} to the given stream.

        @param stream: A file-like object to which this L{Element} will be
            written.

        @param nsprefixes: A C{dict} mapping namespace URIs as C{str} to
            prefixes as C{str}.  This defines the prefixes which are already in
            scope in the document at the point at which this L{Element} exists.
            This is essentially an implementation detail for namespace support.
            Applications should not try to use it.

        @param namespace: The namespace URI as a C{str} which is the default at
            the point in the document at which this L{Element} exists.  This is
            essentially an implementation detail for namespace support.
            Applications should not try to use it.
        