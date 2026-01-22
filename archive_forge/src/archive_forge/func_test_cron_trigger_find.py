from openstack.tests.unit import test_proxy_base
from openstack.workflow.v2 import _proxy
from openstack.workflow.v2 import cron_trigger
from openstack.workflow.v2 import execution
from openstack.workflow.v2 import workflow
def test_cron_trigger_find(self):
    self.verify_find(self.proxy.find_cron_trigger, cron_trigger.CronTrigger, expected_kwargs={'all_projects': False})