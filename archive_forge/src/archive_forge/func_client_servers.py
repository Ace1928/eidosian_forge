import re
import time
from django.core.cache.backends.base import (
from django.utils.functional import cached_property
@property
def client_servers(self):
    output = []
    for server in self._servers:
        output.append(server.removeprefix('unix:'))
    return output