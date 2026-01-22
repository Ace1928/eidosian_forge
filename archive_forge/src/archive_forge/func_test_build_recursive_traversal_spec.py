import collections
from unittest import mock
from oslo_vmware.tests import base
from oslo_vmware import vim_util
@mock.patch.object(vim_util, 'build_selection_spec')
def test_build_recursive_traversal_spec(self, build_selection_spec_mock):
    sel_spec = mock.Mock()
    rp_to_rp_sel_spec = mock.Mock()
    rp_to_vm_sel_spec = mock.Mock()

    def build_sel_spec_side_effect(client_factory, name):
        if name == 'visitFolders':
            return sel_spec
        elif name == 'rp_to_rp':
            return rp_to_rp_sel_spec
        elif name == 'rp_to_vm':
            return rp_to_vm_sel_spec
        else:
            return None
    build_selection_spec_mock.side_effect = build_sel_spec_side_effect
    traversal_spec_dict = {'dc_to_hf': {'type': 'Datacenter', 'path': 'hostFolder', 'skip': False, 'selectSet': [sel_spec]}, 'dc_to_vmf': {'type': 'Datacenter', 'path': 'vmFolder', 'skip': False, 'selectSet': [sel_spec]}, 'dc_to_netf': {'type': 'Datacenter', 'path': 'networkFolder', 'skip': False, 'selectSet': [sel_spec]}, 'dc_to_df': {'type': 'Datacenter', 'path': 'datastoreFolder', 'skip': False, 'selectSet': [sel_spec]}, 'h_to_vm': {'type': 'HostSystem', 'path': 'vm', 'skip': False, 'selectSet': [sel_spec]}, 'cr_to_h': {'type': 'ComputeResource', 'path': 'host', 'skip': False, 'selectSet': []}, 'cr_to_ds': {'type': 'ComputeResource', 'path': 'datastore', 'skip': False, 'selectSet': []}, 'cr_to_rp': {'type': 'ComputeResource', 'path': 'resourcePool', 'skip': False, 'selectSet': [rp_to_rp_sel_spec, rp_to_vm_sel_spec]}, 'ccr_to_h': {'type': 'ClusterComputeResource', 'path': 'host', 'skip': False, 'selectSet': []}, 'ccr_to_ds': {'type': 'ClusterComputeResource', 'path': 'datastore', 'skip': False, 'selectSet': []}, 'ccr_to_rp': {'type': 'ClusterComputeResource', 'path': 'resourcePool', 'skip': False, 'selectSet': [rp_to_rp_sel_spec, rp_to_vm_sel_spec]}, 'rp_to_rp': {'type': 'ResourcePool', 'path': 'resourcePool', 'skip': False, 'selectSet': [rp_to_rp_sel_spec, rp_to_vm_sel_spec]}, 'rp_to_vm': {'type': 'ResourcePool', 'path': 'vm', 'skip': False, 'selectSet': [rp_to_rp_sel_spec, rp_to_vm_sel_spec]}}
    client_factory = mock.Mock()
    client_factory.create.side_effect = lambda ns: mock.Mock()
    trav_spec = vim_util.build_recursive_traversal_spec(client_factory)
    self.assertEqual('visitFolders', trav_spec.name)
    self.assertEqual('childEntity', trav_spec.path)
    self.assertFalse(trav_spec.skip)
    self.assertEqual('Folder', trav_spec.type)
    self.assertEqual(len(traversal_spec_dict) + 1, len(trav_spec.selectSet))
    for spec in trav_spec.selectSet:
        if spec.name not in traversal_spec_dict:
            self.assertEqual(sel_spec, spec)
        else:
            exp_spec = traversal_spec_dict[spec.name]
            self.assertEqual(exp_spec['type'], spec.type)
            self.assertEqual(exp_spec['path'], spec.path)
            self.assertEqual(exp_spec['skip'], spec.skip)
            self.assertEqual(exp_spec['selectSet'], spec.selectSet)