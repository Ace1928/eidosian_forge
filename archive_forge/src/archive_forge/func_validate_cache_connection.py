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
def validate_cache_connection(self):
    if not self._cache_dir:
        raise AnsibleError("error, '%s' cache plugin requires the 'fact_caching_connection' config option to be set (to a writeable directory path)" % self.plugin_name)
    if not os.path.exists(self._cache_dir):
        try:
            os.makedirs(self._cache_dir)
        except (OSError, IOError) as e:
            raise AnsibleError("error in '%s' cache plugin while trying to create cache dir %s : %s" % (self.plugin_name, self._cache_dir, to_bytes(e)))
    else:
        for x in (os.R_OK, os.W_OK, os.X_OK):
            if not os.access(self._cache_dir, x):
                raise AnsibleError("error in '%s' cache, configured path (%s) does not have necessary permissions (rwx), disabling plugin" % (self.plugin_name, self._cache_dir))