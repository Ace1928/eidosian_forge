from __future__ import (absolute_import, division, print_function)
import os
from ansible import constants as C
from ansible import context
from ansible.executor.task_queue_manager import TaskQueueManager, AnsibleEndPlay
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.plugins.loader import become_loader, connection_loader, shell_loader
from ansible.playbook import Playbook
from ansible.template import Templar
from ansible.utils.helpers import pct_to_int
from ansible.utils.collection_loader import AnsibleCollectionConfig
from ansible.utils.collection_loader._collection_finder import _get_collection_name_from_path, _get_collection_playbook_path
from ansible.utils.path import makedirs_safe
from ansible.utils.ssh_functions import set_default_transport
from ansible.utils.display import Display

        Called when a playbook run fails. It generates an inventory which allows
        re-running on ONLY the failed hosts.  This may duplicate some variable
        information in group_vars/host_vars but that is ok, and expected.
        