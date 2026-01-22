from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.mysql.plugins.module_utils.mysql import get_server_version
from ansible_collections.community.mysql.plugins.module_utils.version import LooseVersion
Checks if REPLICA must be used instead of SLAVE