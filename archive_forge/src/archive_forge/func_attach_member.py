from libcloud.common.base import BaseDriver, ConnectionKey
from libcloud.common.types import LibcloudError
def attach_member(self, member):
    return self.driver.balancer_attach_member(balancer=self, member=member)