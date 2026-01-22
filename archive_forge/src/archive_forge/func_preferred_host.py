import logging
import re
from collections import namedtuple
from datetime import time
from urllib.parse import ParseResult, quote, urlparse, urlunparse
@property
def preferred_host(self):
    """Get the preferred host."""
    return self._host