from __future__ import (absolute_import, division, print_function)
import subprocess
from ansible_collections.community.general.plugins.connection.jail import Connection as Jail
from ansible.module_utils.common.text.converters import to_native
from ansible.errors import AnsibleError
from ansible.utils.display import Display
 Local iocage based connections 