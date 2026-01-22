import html
import re
import os
from collections.abc import MutableMapping as DictMixin
from paste import httpexceptions
def not_found_application__set(self, value):
    self.map.not_found_application = value