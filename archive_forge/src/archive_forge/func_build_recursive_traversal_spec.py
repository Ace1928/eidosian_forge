import logging
from oslo_utils import timeutils
from suds import sudsobject
def build_recursive_traversal_spec(client_factory):
    """Builds recursive traversal spec to traverse managed object hierarchy.

    :param client_factory: factory to get API input specs
    :returns: recursive traversal spec
    """
    visit_folders_select_spec = build_selection_spec(client_factory, 'visitFolders')
    dc_to_hf = build_traversal_spec(client_factory, 'dc_to_hf', 'Datacenter', 'hostFolder', False, [visit_folders_select_spec])
    dc_to_vmf = build_traversal_spec(client_factory, 'dc_to_vmf', 'Datacenter', 'vmFolder', False, [visit_folders_select_spec])
    dc_to_netf = build_traversal_spec(client_factory, 'dc_to_netf', 'Datacenter', 'networkFolder', False, [visit_folders_select_spec])
    dc_to_df = build_traversal_spec(client_factory, 'dc_to_df', 'Datacenter', 'datastoreFolder', False, [visit_folders_select_spec])
    h_to_vm = build_traversal_spec(client_factory, 'h_to_vm', 'HostSystem', 'vm', False, [visit_folders_select_spec])
    cr_to_h = build_traversal_spec(client_factory, 'cr_to_h', 'ComputeResource', 'host', False, [])
    cr_to_ds = build_traversal_spec(client_factory, 'cr_to_ds', 'ComputeResource', 'datastore', False, [])
    rp_to_rp_select_spec = build_selection_spec(client_factory, 'rp_to_rp')
    rp_to_vm_select_spec = build_selection_spec(client_factory, 'rp_to_vm')
    cr_to_rp = build_traversal_spec(client_factory, 'cr_to_rp', 'ComputeResource', 'resourcePool', False, [rp_to_rp_select_spec, rp_to_vm_select_spec])
    ccr_to_h = build_traversal_spec(client_factory, 'ccr_to_h', 'ClusterComputeResource', 'host', False, [])
    ccr_to_ds = build_traversal_spec(client_factory, 'ccr_to_ds', 'ClusterComputeResource', 'datastore', False, [])
    ccr_to_rp = build_traversal_spec(client_factory, 'ccr_to_rp', 'ClusterComputeResource', 'resourcePool', False, [rp_to_rp_select_spec, rp_to_vm_select_spec])
    rp_to_rp = build_traversal_spec(client_factory, 'rp_to_rp', 'ResourcePool', 'resourcePool', False, [rp_to_rp_select_spec, rp_to_vm_select_spec])
    rp_to_vm = build_traversal_spec(client_factory, 'rp_to_vm', 'ResourcePool', 'vm', False, [rp_to_rp_select_spec, rp_to_vm_select_spec])
    traversal_spec = build_traversal_spec(client_factory, 'visitFolders', 'Folder', 'childEntity', False, [visit_folders_select_spec, h_to_vm, dc_to_hf, dc_to_vmf, dc_to_netf, dc_to_df, cr_to_ds, cr_to_h, cr_to_rp, ccr_to_h, ccr_to_ds, ccr_to_rp, rp_to_rp, rp_to_vm])
    return traversal_spec