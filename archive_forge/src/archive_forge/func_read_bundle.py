import base64
import re
from io import BytesIO
from .... import errors, registry
from ....diff import internal_diff
from ....revision import NULL_REVISION
from ....timestamp import format_highres_date, unpack_highres_date
def read_bundle(f):
    """Read in a bundle from a filelike object.

    :param f: A file-like object
    :return: A list of Bundle objects
    """
    version = None
    for line in f:
        m = BUNDLE_HEADER_RE.match(line)
        if m:
            if m.group('lineending') != b'':
                raise errors.UnsupportedEOLMarker()
            version = m.group('version')
            break
        elif line.startswith(BUNDLE_HEADER):
            raise errors.MalformedHeader('Extra characters after version number')
        m = CHANGESET_OLD_HEADER_RE.match(line)
        if m:
            version = m.group('version')
            raise errors.BundleNotSupported(version, 'old format bundles not supported')
    if version is None:
        raise errors.NotABundle('Did not find an opening header')
    return get_serializer(version.decode('ascii')).read(f)