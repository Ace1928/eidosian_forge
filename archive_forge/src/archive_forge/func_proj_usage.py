from oslotest import base
from oslo_config import cfg
from oslo_config import fixture as config_fixture
from oslo_limit import exception
from oslo_limit import fixture
from oslo_limit import limit
from oslo_limit import opts
def proj_usage(project_id, resource_names):
    return self.usage[project_id]