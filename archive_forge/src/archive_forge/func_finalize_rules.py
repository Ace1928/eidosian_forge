import logging
import re
from collections import namedtuple
from datetime import time
from urllib.parse import ParseResult, quote, urlparse, urlunparse
def finalize_rules(self):
    self._rules.sort(key=lambda r: (r.value.priority, r.field == 'allow'), reverse=True)