from webob import exc
from heat.api.openstack.v1 import util
from heat.common import serializers
from heat.common import wsgi
from heat.rpc import client as rpc_client
Delete an existing software deployment.