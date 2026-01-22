from openstack import exceptions
from heat.common import exception
from heat.common.i18n import _
from heat.engine.clients.os import openstacksdk as sdk_plugin
from heat.engine import constraints
def get_cluster_id(self, cluster_name):
    cluster = self.client().get_cluster(cluster_name)
    return cluster.id