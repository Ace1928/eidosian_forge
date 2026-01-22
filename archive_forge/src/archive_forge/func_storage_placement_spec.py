import logging
from oslo_utils import timeutils
from suds import sudsobject
def storage_placement_spec(client_factory, dsc_ref, type, clone_spec=None, config_spec=None, relocate_spec=None, vm_ref=None, folder=None, clone_name=None, res_pool_ref=None, host_ref=None):
    pod_sel_spec = client_factory.create('ns0:StorageDrsPodSelectionSpec')
    pod_sel_spec.storagePod = dsc_ref
    spec = client_factory.create('ns0:StoragePlacementSpec')
    spec.podSelectionSpec = pod_sel_spec
    spec.type = type
    spec.vm = vm_ref
    spec.folder = folder
    spec.cloneSpec = clone_spec
    spec.configSpec = config_spec
    spec.relocateSpec = relocate_spec
    spec.cloneName = clone_name
    spec.resourcePool = res_pool_ref
    spec.host = host_ref
    return spec