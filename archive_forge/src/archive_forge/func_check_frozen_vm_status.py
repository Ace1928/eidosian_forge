import copy
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import RLock
from typing import Any, Dict
import com.vmware.vapi.std.errors_client as ErrorClients
from com.vmware.cis.tagging_client import CategoryModel
from com.vmware.content.library_client import Item
from com.vmware.vapi.std_client import DynamicID
from com.vmware.vcenter.ovf_client import DiskProvisioningType, LibraryItem
from com.vmware.vcenter.vm.hardware_client import Cpu, Memory
from com.vmware.vcenter.vm_client import Power as HardPower
from com.vmware.vcenter_client import VM, Host, ResourcePool
from pyVim.task import WaitForTask
from pyVmomi import vim
from ray.autoscaler._private.cli_logger import cli_logger
from ray.autoscaler._private.vsphere.config import (
from ray.autoscaler._private.vsphere.gpu_utils import (
from ray.autoscaler._private.vsphere.pyvmomi_sdk_provider import PyvmomiSdkProvider
from ray.autoscaler._private.vsphere.scheduler import SchedulerFactory
from ray.autoscaler._private.vsphere.utils import Constants, is_ipv4, now_ts
from ray.autoscaler._private.vsphere.vsphere_sdk_provider import VsphereSdkProvider
from ray.autoscaler.node_provider import NodeProvider
from ray.autoscaler.tags import TAG_RAY_CLUSTER_NAME, TAG_RAY_NODE_NAME
def check_frozen_vm_status(self, frozen_vm_name):
    """
        This function will help check if the frozen VM with the specific name is
        existing and in the frozen state. If the frozen VM is existing and off, this
        function will also help to power on the frozen VM and wait until it is frozen.
        """
    vm = self.get_pyvmomi_sdk_provider().get_pyvmomi_obj([vim.VirtualMachine], frozen_vm_name)
    if vm.runtime.powerState == vim.VirtualMachinePowerState.poweredOff:
        logger.debug(f'Frozen VM {vm._moId} is off. Powering it ON')
        WaitForTask(vm.PowerOnVM_Task())
    wait_until_vm_is_frozen(vm)
    return vm