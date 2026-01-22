from libcloud.common.base import BaseDriver, ConnectionKey
from libcloud.common.types import LibcloudError
def attach_compute_node(self, node):
    return self.driver.balancer_attach_compute_node(balancer=self, node=node)