import logging
import re
from collections import namedtuple
from datetime import time
from urllib.parse import ParseResult, quote, urlparse, urlunparse
def hex_to_byte(h):
    """Replaces a %xx escape with equivalent binary sequence."""
    return bytes.fromhex(h)