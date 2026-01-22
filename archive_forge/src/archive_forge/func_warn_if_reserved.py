from __future__ import (absolute_import, division, print_function)
from ansible.playbook import Play
from ansible.playbook.block import Block
from ansible.playbook.role import Role
from ansible.playbook.task import Task
from ansible.utils.display import Display
def warn_if_reserved(myvars, additional=None):
    """ this function warns if any variable passed conflicts with internally reserved names """
    if additional is None:
        reserved = _RESERVED_NAMES
    else:
        reserved = _RESERVED_NAMES.union(additional)
    varnames = set(myvars)
    varnames.discard('vars')
    for varname in varnames.intersection(reserved):
        display.warning('Found variable using reserved name: %s' % varname)