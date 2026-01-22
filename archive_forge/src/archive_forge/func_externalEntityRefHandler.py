import os
import warnings
from collections import Counter
from xml.parsers import expat
from io import BytesIO
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape
from urllib.request import urlopen
from urllib.parse import urlparse
from Bio import StreamModeError
def externalEntityRefHandler(self, context, base, systemId, publicId):
    """Handle external entity reference in order to cache DTD locally.

        The purpose of this function is to load the DTD locally, instead
        of downloading it from the URL specified in the XML. Using the local
        DTD results in much faster parsing. If the DTD is not found locally,
        we try to download it. If new DTDs become available from NCBI,
        putting them in Bio/Entrez/DTDs will allow the parser to see them.
        """
    urlinfo = urlparse(systemId)
    if urlinfo.scheme in ['http', 'https', 'ftp']:
        url = systemId
    elif urlinfo.scheme == '':
        try:
            source = self.dtd_urls[-1]
        except IndexError:
            source = 'http://www.ncbi.nlm.nih.gov/dtd/'
        else:
            source = os.path.dirname(source)
        url = source.rstrip('/') + '/' + systemId
    else:
        raise ValueError('Unexpected URL scheme %r' % urlinfo.scheme)
    self.dtd_urls.append(url)
    location, filename = os.path.split(systemId)
    handle = self.open_dtd_file(filename)
    if not handle:
        try:
            handle = urlopen(url)
        except OSError:
            raise RuntimeError(f'Failed to access {filename} at {url}') from None
        text = handle.read()
        handle.close()
        self.save_dtd_file(filename, text)
        handle = BytesIO(text)
    parser = self.parser.ExternalEntityParserCreate(context)
    parser.ElementDeclHandler = self.elementDecl
    parser.ParseFile(handle)
    handle.close()
    self.dtd_urls.pop()
    self.parser.StartElementHandler = self.startElementHandler
    return 1