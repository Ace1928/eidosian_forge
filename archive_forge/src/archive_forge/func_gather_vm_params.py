from __future__ import absolute_import, division, print_function
import atexit
import time
import re
import traceback
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils.ansible_release import __version__ as ANSIBLE_VERSION
def gather_vm_params(module, vm_ref):
    """Gathers all VM parameters available in XAPI database.

    Args:
        module: Reference to Ansible module object.
        vm_ref (str): XAPI reference to VM.

    Returns:
        dict: VM parameters.
    """
    if not vm_ref or vm_ref == 'OpaqueRef:NULL':
        return {}
    xapi_session = XAPI.connect(module)
    try:
        vm_params = xapi_session.xenapi.VM.get_record(vm_ref)
        if vm_params['affinity'] != 'OpaqueRef:NULL':
            vm_affinity = xapi_session.xenapi.host.get_record(vm_params['affinity'])
            vm_params['affinity'] = vm_affinity
        else:
            vm_params['affinity'] = {}
        vm_vbd_params_list = [xapi_session.xenapi.VBD.get_record(vm_vbd_ref) for vm_vbd_ref in vm_params['VBDs']]
        vm_vbd_params_list = sorted(vm_vbd_params_list, key=lambda vm_vbd_params: int(vm_vbd_params['userdevice']))
        vm_params['VBDs'] = vm_vbd_params_list
        for vm_vbd_params in vm_params['VBDs']:
            if vm_vbd_params['VDI'] != 'OpaqueRef:NULL':
                vm_vdi_params = xapi_session.xenapi.VDI.get_record(vm_vbd_params['VDI'])
            else:
                vm_vdi_params = {}
            vm_vbd_params['VDI'] = vm_vdi_params
        vm_vif_params_list = [xapi_session.xenapi.VIF.get_record(vm_vif_ref) for vm_vif_ref in vm_params['VIFs']]
        vm_vif_params_list = sorted(vm_vif_params_list, key=lambda vm_vif_params: int(vm_vif_params['device']))
        vm_params['VIFs'] = vm_vif_params_list
        for vm_vif_params in vm_params['VIFs']:
            if vm_vif_params['network'] != 'OpaqueRef:NULL':
                vm_network_params = xapi_session.xenapi.network.get_record(vm_vif_params['network'])
            else:
                vm_network_params = {}
            vm_vif_params['network'] = vm_network_params
        if vm_params['guest_metrics'] != 'OpaqueRef:NULL':
            vm_guest_metrics = xapi_session.xenapi.VM_guest_metrics.get_record(vm_params['guest_metrics'])
            vm_params['guest_metrics'] = vm_guest_metrics
        else:
            vm_params['guest_metrics'] = {}
        xenserver_version = get_xenserver_version(module)
        if xenserver_version[0] >= 7 and xenserver_version[1] >= 0 and vm_params.get('guest_metrics') and ('feature-static-ip-setting' in vm_params['guest_metrics']['other']):
            vm_params['customization_agent'] = 'native'
        else:
            vm_params['customization_agent'] = 'custom'
    except XenAPI.Failure as f:
        module.fail_json(msg='XAPI ERROR: %s' % f.details)
    return vm_params