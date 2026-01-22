from openstack import exceptions
from openstack.tests.functional.baremetal import base
def test_baremetal_deploy_update(self):
    steps = [{'interface': 'bios', 'step': 'apply_configuration', 'args': {'settings': [{'name': 'LogicalProc', 'value': 'Enabled'}]}, 'priority': 150}]
    deploy_template = self.create_deploy_template(name='CUSTOM_DEPLOY_TEMPLATE4', steps=steps)
    deploy_template.extra = {'answer': 42}
    deploy_template = self.conn.baremetal.update_deploy_template(deploy_template)
    self.assertEqual({'answer': 42}, deploy_template.extra)
    deploy_template = self.conn.baremetal.get_deploy_template(deploy_template.id)
    self.assertEqual({'answer': 42}, deploy_template.extra)