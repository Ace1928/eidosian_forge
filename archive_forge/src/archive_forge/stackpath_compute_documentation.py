from __future__ import (absolute_import, division, print_function)
import traceback
import json
from ansible.errors import AnsibleError
from ansible.module_utils.urls import open_url
from ansible.plugins.inventory import (
from ansible.utils.display import Display
from ansible.utils.unsafe_proxy import wrap_var as make_unsafe

            :param loader: an ansible.parsing.dataloader.DataLoader object
            :param path: the path to the inventory config file
            :return the contents of the config file
        