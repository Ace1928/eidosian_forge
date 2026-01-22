from unittest import mock
from oslo_config import cfg
import swiftclient.client as sc
from heat.common import exception
from heat.common import template_format
from heat.engine.resources.aws.s3 import s3
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_website(self):
    t = template_format.parse(swift_template)
    stack = utils.parse_stack(t)
    container_name = utils.PhysName(stack.name, 'test_resource')
    self.mock_con.put_container.return_value = None
    self.mock_con.delete_container.return_value = None
    rsrc = self.create_resource(t, stack, 'S3BucketWebsite')
    scheduler.TaskRunner(rsrc.delete)()
    self.mock_con.put_container.assert_called_once_with(container_name, {'X-Container-Meta-Web-Error': 'error.html', 'X-Container-Meta-Web-Index': 'index.html', 'X-Container-Write': 'test_tenant:test_username', 'X-Container-Read': '.r:*'})
    self.mock_con.delete_container.assert_called_once_with(container_name)