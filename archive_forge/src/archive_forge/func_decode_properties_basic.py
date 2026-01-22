import calendar
from datetime import datetime
from decimal import Decimal
from io import BytesIO
from struct import pack, unpack_from
from .exceptions import FrameSyntaxError
from .spec import Basic
from .utils import bytes_to_str as pstr_t
from .utils import str_to_bytes
def decode_properties_basic(buf, offset):
    """Decode basic properties."""
    properties = {}
    flags, = unpack_from('>H', buf, offset)
    offset += 2
    if flags & 32768:
        slen, = unpack_from('>B', buf, offset)
        offset += 1
        properties['content_type'] = pstr_t(buf[offset:offset + slen])
        offset += slen
    if flags & 16384:
        slen, = unpack_from('>B', buf, offset)
        offset += 1
        properties['content_encoding'] = pstr_t(buf[offset:offset + slen])
        offset += slen
    if flags & 8192:
        _f, offset = loads('F', buf, offset)
        properties['application_headers'], = _f
    if flags & 4096:
        properties['delivery_mode'], = unpack_from('>B', buf, offset)
        offset += 1
    if flags & 2048:
        properties['priority'], = unpack_from('>B', buf, offset)
        offset += 1
    if flags & 1024:
        slen, = unpack_from('>B', buf, offset)
        offset += 1
        properties['correlation_id'] = pstr_t(buf[offset:offset + slen])
        offset += slen
    if flags & 512:
        slen, = unpack_from('>B', buf, offset)
        offset += 1
        properties['reply_to'] = pstr_t(buf[offset:offset + slen])
        offset += slen
    if flags & 256:
        slen, = unpack_from('>B', buf, offset)
        offset += 1
        properties['expiration'] = pstr_t(buf[offset:offset + slen])
        offset += slen
    if flags & 128:
        slen, = unpack_from('>B', buf, offset)
        offset += 1
        properties['message_id'] = pstr_t(buf[offset:offset + slen])
        offset += slen
    if flags & 64:
        properties['timestamp'], = unpack_from('>Q', buf, offset)
        offset += 8
    if flags & 32:
        slen, = unpack_from('>B', buf, offset)
        offset += 1
        properties['type'] = pstr_t(buf[offset:offset + slen])
        offset += slen
    if flags & 16:
        slen, = unpack_from('>B', buf, offset)
        offset += 1
        properties['user_id'] = pstr_t(buf[offset:offset + slen])
        offset += slen
    if flags & 8:
        slen, = unpack_from('>B', buf, offset)
        offset += 1
        properties['app_id'] = pstr_t(buf[offset:offset + slen])
        offset += slen
    if flags & 4:
        slen, = unpack_from('>B', buf, offset)
        offset += 1
        properties['cluster_id'] = pstr_t(buf[offset:offset + slen])
        offset += slen
    return (properties, offset)