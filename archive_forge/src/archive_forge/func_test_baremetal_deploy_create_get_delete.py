from openstack import exceptions
from openstack.tests.functional.baremetal import base
def test_baremetal_deploy_create_get_delete(self):
    steps = [{'interface': 'bios', 'step': 'apply_configuration', 'args': {'settings': [{'name': 'LogicalProc', 'value': 'Enabled'}]}, 'priority': 150}]
    deploy_template = self.create_deploy_template(name='CUSTOM_DEPLOY_TEMPLATE', steps=steps)
    loaded = self.conn.baremetal.get_deploy_template(deploy_template.id)
    self.assertEqual(loaded.id, deploy_template.id)
    self.conn.baremetal.delete_deploy_template(deploy_template, ignore_missing=False)
    self.assertRaises(exceptions.ResourceNotFound, self.conn.baremetal.get_deploy_template, deploy_template.id)