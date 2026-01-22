from __future__ import absolute_import, division, print_function
from abc import ABCMeta, abstractmethod
from ansible.module_utils.six import with_metaclass
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.common._utils import get_all_subclasses
@abstractmethod
def get_package_details(self, package):
    pass