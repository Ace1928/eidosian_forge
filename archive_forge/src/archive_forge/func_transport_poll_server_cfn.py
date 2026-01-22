import uuid
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from heat.common import exception
from heat.common import password_gen
from heat.engine.clients import progress
from heat.engine.resources import stack_user
def transport_poll_server_cfn(self, props):
    return props[self.SOFTWARE_CONFIG_TRANSPORT] == self.POLL_SERVER_CFN