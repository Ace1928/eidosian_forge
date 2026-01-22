from __future__ import (absolute_import, division, print_function)
import copy
import errno
import os
import tempfile
import time
from abc import abstractmethod
from collections.abc import MutableMapping
from ansible import constants as C
from ansible.errors import AnsibleError
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.plugins import AnsiblePlugin
from ansible.plugins.loader import cache_loader
from ansible.utils.collection_loader import resource_from_fqcr
from ansible.utils.display import Display
def update_cache_if_changed(self):
    if self._retrieved != self._cache:
        self.set_cache()