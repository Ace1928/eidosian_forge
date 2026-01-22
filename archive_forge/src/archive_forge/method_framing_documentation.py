from collections import defaultdict
from struct import pack, pack_into, unpack_from
from . import spec
from .basic_message import Message
from .exceptions import UnexpectedFrame
from .utils import str_to_bytes
Create closure that writes frames.