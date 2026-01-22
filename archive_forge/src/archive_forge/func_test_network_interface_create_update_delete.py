import copy
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_network_interface_create_update_delete(self):
    my_stack = utils.parse_stack(test_template, stack_name='test_nif_cud_stack')
    nic_rsrc = my_stack['my_nic']
    self.mock_show_subnet()
    self.stub_SubnetConstraint_validate()
    self.mock_create_network_interface(my_stack.name)
    update_props = {}
    update_sg_ids = ['0389f747-7785-4757-b7bb-2ab07e4b09c3']
    update_props['security_groups'] = update_sg_ids
    self.assertIsNone(nic_rsrc.validate())
    scheduler.TaskRunner(nic_rsrc.create)()
    self.assertEqual((nic_rsrc.CREATE, my_stack.COMPLETE), nic_rsrc.state)
    props = copy.deepcopy(nic_rsrc.properties.data)
    props['GroupSet'] = update_sg_ids
    update_snippet = rsrc_defn.ResourceDefinition(nic_rsrc.name, nic_rsrc.type(), props)
    scheduler.TaskRunner(nic_rsrc.update, update_snippet)()
    self.assertEqual((nic_rsrc.UPDATE, nic_rsrc.COMPLETE), nic_rsrc.state)
    scheduler.TaskRunner(nic_rsrc.delete)()
    self.assertEqual((nic_rsrc.DELETE, nic_rsrc.COMPLETE), nic_rsrc.state)
    self.m_ss.assert_called_once_with('ssss')
    self.m_cp.assert_called_once_with({'port': self.port})
    self.m_up.assert_called_once_with('pppp', {'port': update_props})
    self.m_dp.assert_called_once_with('pppp')