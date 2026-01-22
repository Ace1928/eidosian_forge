from libcloud.common.base import BaseDriver, ConnectionKey
from libcloud.common.types import LibcloudError
def detach_member(self, member):
    return self.driver.balancer_detach_member(balancer=self, member=member)