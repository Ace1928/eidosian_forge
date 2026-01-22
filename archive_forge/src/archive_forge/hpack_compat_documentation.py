import logging
from .hpack import _to_bytes

            Takes an HPACK-encoded header block and decodes it into a header
            set.
            