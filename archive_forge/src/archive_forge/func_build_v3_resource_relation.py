from oslo_serialization import jsonutils
from keystone import exception
from keystone.i18n import _
def build_v3_resource_relation(resource_name):
    return 'https://docs.openstack.org/api/openstack-identity/3/rel/%s' % resource_name