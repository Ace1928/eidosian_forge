from __future__ import absolute_import, division, print_function
from abc import ABCMeta, abstractmethod
from ansible.module_utils.six import with_metaclass
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.common._utils import get_all_subclasses
def get_all_pkg_managers():
    return {obj.__name__.lower(): obj for obj in get_all_subclasses(PkgMgr) if obj not in (CLIMgr, LibMgr)}