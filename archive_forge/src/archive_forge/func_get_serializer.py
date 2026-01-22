import base64
import re
from io import BytesIO
from .... import errors, registry
from ....diff import internal_diff
from ....revision import NULL_REVISION
from ....timestamp import format_highres_date, unpack_highres_date
def get_serializer(version):
    try:
        serializer = serializer_registry.get(version)
    except KeyError:
        raise errors.BundleNotSupported(version, 'unknown bundle format')
    return serializer(version)