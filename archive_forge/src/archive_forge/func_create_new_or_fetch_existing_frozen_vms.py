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
def create_new_or_fetch_existing_frozen_vms(self, node_config):
    frozen_vm_obj = None
    frozen_vm_config = node_config['frozen_vm']
    frozen_vm_resource_pool = frozen_vm_config.get('resource_pool')
    rp_obj = None
    if frozen_vm_resource_pool:
        rp_obj = self.get_pyvmomi_sdk_provider().get_pyvmomi_obj([vim.ResourcePool], frozen_vm_resource_pool)
        if not self.frozen_vm_scheduler:
            self.frozen_vm_scheduler = SchedulerFactory.get_scheduler(rp_obj)
    if frozen_vm_config.get('library_item'):
        if frozen_vm_resource_pool:
            self.create_frozen_vm_on_each_host(node_config, frozen_vm_config.get('name', 'frozen-vm'), rp_obj)
            frozen_vm_obj = self.frozen_vm_scheduler.next_frozen_vm()
        else:
            frozen_vm_obj = self.create_frozen_vm_from_ovf(node_config, frozen_vm_config['name'])
    elif frozen_vm_resource_pool:
        self.check_frozen_vms_status(rp_obj)
        frozen_vm_obj = self.frozen_vm_scheduler.next_frozen_vm()
    else:
        frozen_vm_name = frozen_vm_config.get('name', 'frozen-vm')
        frozen_vm_obj = self.check_frozen_vm_status(frozen_vm_name)
    return frozen_vm_obj