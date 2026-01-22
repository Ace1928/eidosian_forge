from boto.ec2.elb.healthcheck import HealthCheck
from boto.ec2.elb.listener import Listener
from boto.ec2.elb.listelement import ListElement
from boto.ec2.elb.policies import Policies, OtherPolicy
from boto.ec2.elb.securitygroup import SecurityGroup
from boto.ec2.instanceinfo import InstanceInfo
from boto.resultset import ResultSet
from boto.compat import six
def is_cross_zone_load_balancing(self, force=False):
    """
        Identifies if the ELB is current configured to do CrossZone Balancing.

        :type force: bool
        :param force: Ignore cache value and reload.

        :rtype: bool
        :return: True if balancing is enabled, False if not.
        """
    return self.get_attributes(force).cross_zone_load_balancing.enabled