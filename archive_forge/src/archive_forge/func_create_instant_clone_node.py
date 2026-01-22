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
def create_instant_clone_node(self, source_vm, vm_name_target, node_config, tags, gpu_cards_map):
    resource_pool = self.get_pyvmomi_sdk_provider().get_pyvmomi_obj([vim.ResourcePool], node_config.get('resource_pool')) if node_config.get('resource_pool') else None
    datastore = self.get_pyvmomi_sdk_provider().get_pyvmomi_obj([vim.Datastore], node_config.get('datastore', None)) if node_config.get('datastore') else None
    resources = node_config['resources']
    vm_relocate_spec = vim.vm.RelocateSpec(pool=resource_pool, datastore=datastore)
    instant_clone_spec = vim.vm.InstantCloneSpec(name=vm_name_target, location=vm_relocate_spec)
    to_be_plugged_gpu = []
    parent_vm = None
    requested_gpu_num = resources.get('GPU', 0)
    if requested_gpu_num > 0:
        if not gpu_cards_map:
            raise ValueError(f'No available GPU card to assigned to node {vm_name_target}')
        for vm_name in gpu_cards_map:
            parent_vm = self.get_pyvmomi_sdk_provider().get_pyvmomi_obj([vim.VirtualMachine], vm_name)
            to_be_plugged_gpu = gpu_cards_map[vm_name]
            break
    else:
        parent_vm = source_vm
    tags[Constants.VSPHERE_NODE_STATUS] = Constants.VsphereNodeStatus.CREATING.value
    threading.Thread(target=self.tag_vm, args=(vm_name_target, tags)).start()
    WaitForTask(parent_vm.InstantClone_Task(spec=instant_clone_spec))
    logger.info(f'Clone VM {vm_name_target} from Frozen-VM {parent_vm.name}')
    cloned_vm = self.get_pyvmomi_sdk_provider().get_pyvmomi_obj([vim.VirtualMachine], vm_name_target)
    vm = self.pyvmomi_vm_to_vsphere_sdk_vm(cloned_vm)
    if 'CPU' in resources:
        update_spec = Cpu.UpdateSpec(count=resources['CPU'])
        logger.debug('vm.hardware.Cpu.update({}, {})'.format(cloned_vm.name, update_spec))
        self.get_vsphere_sdk_client().vcenter.vm.hardware.Cpu.update(vm.vm, update_spec)
    if 'Memory' in resources:
        update_spec = Memory.UpdateSpec(size_mib=resources['Memory'])
        logger.debug('vm.hardware.Memory.update({}, {})'.format(cloned_vm.name, update_spec))
        self.get_vsphere_sdk_client().vcenter.vm.hardware.Memory.update(vm.vm, update_spec)
    if to_be_plugged_gpu:
        is_dynamic = is_dynamic_passthrough(node_config)
        add_gpus_to_vm(self.get_pyvmomi_sdk_provider(), cloned_vm.name, to_be_plugged_gpu, is_dynamic)
    return vm