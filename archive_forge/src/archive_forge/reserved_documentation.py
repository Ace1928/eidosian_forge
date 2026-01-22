from __future__ import (absolute_import, division, print_function)
from ansible.playbook import Play
from ansible.playbook.block import Block
from ansible.playbook.role import Role
from ansible.playbook.task import Task
from ansible.utils.display import Display
 this function warns if any variable passed conflicts with internally reserved names 