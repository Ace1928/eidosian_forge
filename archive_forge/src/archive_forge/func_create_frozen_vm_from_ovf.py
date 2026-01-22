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
def create_frozen_vm_from_ovf(self, node_config, vm_name_target):
    resource_pool_id = None
    datastore_name = node_config.get('frozen_vm').get('datastore')
    if not datastore_name:
        raise ValueError('The datastore name must be provided when deploying frozenVM from OVF')
    datastore_mo = self.get_pyvmomi_sdk_provider().get_pyvmomi_obj([vim.Datastore], datastore_name)
    datastore_id = datastore_mo._moId
    if node_config.get('frozen_vm').get('resource_pool'):
        rp_filter_spec = ResourcePool.FilterSpec(names={node_config['frozen_vm']['resource_pool']})
        resource_pool_summaries = self.get_vsphere_sdk_client().vcenter.ResourcePool.list(rp_filter_spec)
        if not resource_pool_summaries:
            raise ValueError("Resource pool with name '{}' not found".format(rp_filter_spec))
        resource_pool_id = resource_pool_summaries[0].resource_pool
        logger.debug('Resource pool ID: {}'.format(resource_pool_id))
    else:
        cluster_name = node_config.get('frozen_vm').get('cluster')
        if not cluster_name:
            raise ValueError('The cluster name must be provided when deploying a single frozen VM from OVF')
        cluster_mo = self.get_pyvmomi_sdk_provider().get_pyvmomi_obj([vim.ClusterComputeResource], cluster_name)
        node_config['host_id'] = cluster_mo.host[0]._moId
        resource_pool_id = cluster_mo.resourcePool._moId
    lib_item = node_config['frozen_vm']['library_item']
    find_spec = Item.FindSpec(name=lib_item)
    item_ids = self.get_vsphere_sdk_client().content.library.Item.find(find_spec)
    if len(item_ids) < 1:
        raise ValueError("Content library items with name '{}' not found".format(lib_item))
    if len(item_ids) > 1:
        logger.warning("Unexpected: found multiple content library items with name                 '{}'".format(lib_item))
    lib_item_id = item_ids[0]
    deployment_target = LibraryItem.DeploymentTarget(resource_pool_id=resource_pool_id, host_id=node_config.get('host_id'))
    ovf_summary = self.get_vsphere_sdk_client().vcenter.ovf.LibraryItem.filter(ovf_library_item_id=lib_item_id, target=deployment_target)
    logger.info('Found an OVF template: {} to deploy.'.format(ovf_summary.name))
    deployment_spec = LibraryItem.ResourcePoolDeploymentSpec(name=vm_name_target, annotation=ovf_summary.annotation, accept_all_eula=True, network_mappings=None, storage_mappings=None, storage_provisioning=DiskProvisioningType.thin, storage_profile_id=None, locale=None, flags=None, additional_parameters=None, default_datastore_id=datastore_id)
    result = self.get_vsphere_sdk_client().vcenter.ovf.LibraryItem.deploy(lib_item_id, deployment_target, deployment_spec, client_token=str(uuid.uuid4()))
    logger.debug('result: {}'.format(result))
    if len(result.error.errors) > 0:
        for error in result.error.errors:
            logger.error('OVF error: {}'.format(error))
        raise ValueError('OVF deployment failed for VM {}, reason: {}'.format(vm_name_target, result))
    logger.info('Deployment successful. VM Name: "{}", ID: "{}"'.format(vm_name_target, result.resource_id.id))
    error = result.error
    if error is not None:
        for warning in error.warnings:
            logger.warning('OVF warning: {}'.format(warning.message))
    vm_id = result.resource_id.id
    vm = self.get_vsphere_sdk_vm_obj(vm_id)
    pyvmomi_vm_obj = self.check_frozen_vm_status(vm.name)
    return pyvmomi_vm_obj