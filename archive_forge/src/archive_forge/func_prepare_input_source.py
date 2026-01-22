import os, urllib.parse, urllib.request
import io
import codecs
from . import handler
from . import xmlreader
def prepare_input_source(source, base=''):
    """This function takes an InputSource and an optional base URL and
    returns a fully resolved InputSource object ready for reading."""
    if isinstance(source, os.PathLike):
        source = os.fspath(source)
    if isinstance(source, str):
        source = xmlreader.InputSource(source)
    elif hasattr(source, 'read'):
        f = source
        source = xmlreader.InputSource()
        if isinstance(f.read(0), str):
            source.setCharacterStream(f)
        else:
            source.setByteStream(f)
        if hasattr(f, 'name') and isinstance(f.name, str):
            source.setSystemId(f.name)
    if source.getCharacterStream() is None and source.getByteStream() is None:
        sysid = source.getSystemId()
        basehead = os.path.dirname(os.path.normpath(base))
        sysidfilename = os.path.join(basehead, sysid)
        if os.path.isfile(sysidfilename):
            source.setSystemId(sysidfilename)
            f = open(sysidfilename, 'rb')
        else:
            source.setSystemId(urllib.parse.urljoin(base, sysid))
            f = urllib.request.urlopen(source.getSystemId())
        source.setByteStream(f)
    return source