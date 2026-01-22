from __future__ import absolute_import, division, print_function
import re
import time
import string
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.network import is_mac
from ansible.module_utils._text import to_text, to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible_collections.community.vmware.plugins.module_utils.vm_device_helper import PyVmomiDeviceHelper
from ansible_collections.community.vmware.plugins.module_utils.vmware_spbm import SPBM
def configure_resource_alloc_info(self, vm_obj):
    """
        Function to configure resource allocation information about virtual machine
        :param vm_obj: VM object in case of reconfigure, None in case of deploy
        :return: None
        """
    rai_change_detected = False
    memory_allocation = vim.ResourceAllocationInfo()
    cpu_allocation = vim.ResourceAllocationInfo()
    memory_shares_info = vim.SharesInfo()
    cpu_shares_info = vim.SharesInfo()
    mem_shares_level = self.params['hardware']['mem_shares_level']
    if mem_shares_level is not None:
        memory_shares_info.level = mem_shares_level
        memory_allocation.shares = memory_shares_info
        if vm_obj is None or memory_allocation.shares.level != vm_obj.config.memoryAllocation.shares.level:
            rai_change_detected = True
    cpu_shares_level = self.params['hardware']['cpu_shares_level']
    if cpu_shares_level is not None:
        cpu_shares_info.level = cpu_shares_level
        cpu_allocation.shares = cpu_shares_info
        if vm_obj is None or cpu_allocation.shares.level != vm_obj.config.cpuAllocation.shares.level:
            rai_change_detected = True
    mem_shares = self.params['hardware']['mem_shares']
    if mem_shares is not None:
        memory_shares_info.level = 'custom'
        memory_shares_info.shares = mem_shares
        memory_allocation.shares = memory_shares_info
        if vm_obj is None or memory_allocation.shares != vm_obj.config.memoryAllocation.shares:
            rai_change_detected = True
    cpu_shares = self.params['hardware']['cpu_shares']
    if cpu_shares is not None:
        cpu_shares_info.level = 'custom'
        cpu_shares_info.shares = cpu_shares
        cpu_allocation.shares = cpu_shares_info
        if vm_obj is None or cpu_allocation.shares != vm_obj.config.cpuAllocation.shares:
            rai_change_detected = True
    mem_limit = self.params['hardware']['mem_limit']
    if mem_limit is not None:
        memory_allocation.limit = mem_limit
        if vm_obj is None or memory_allocation.limit != vm_obj.config.memoryAllocation.limit:
            rai_change_detected = True
    mem_reservation = self.params['hardware']['mem_reservation']
    if mem_reservation is not None:
        memory_allocation.reservation = mem_reservation
        if vm_obj is None or memory_allocation.reservation != vm_obj.config.memoryAllocation.reservation:
            rai_change_detected = True
    cpu_limit = self.params['hardware']['cpu_limit']
    if cpu_limit is not None:
        cpu_allocation.limit = cpu_limit
        if vm_obj is None or cpu_allocation.limit != vm_obj.config.cpuAllocation.limit:
            rai_change_detected = True
    cpu_reservation = self.params['hardware']['cpu_reservation']
    if cpu_reservation is not None:
        cpu_allocation.reservation = cpu_reservation
        if vm_obj is None or cpu_allocation.reservation != vm_obj.config.cpuAllocation.reservation:
            rai_change_detected = True
    if rai_change_detected:
        self.configspec.memoryAllocation = memory_allocation
        self.configspec.cpuAllocation = cpu_allocation
        self.change_detected = True