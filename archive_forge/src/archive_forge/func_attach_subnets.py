from boto.ec2.elb.healthcheck import HealthCheck
from boto.ec2.elb.listener import Listener
from boto.ec2.elb.listelement import ListElement
from boto.ec2.elb.policies import Policies, OtherPolicy
from boto.ec2.elb.securitygroup import SecurityGroup
from boto.ec2.instanceinfo import InstanceInfo
from boto.resultset import ResultSet
from boto.compat import six
def attach_subnets(self, subnets):
    """
        Attaches load balancer to one or more subnets.
        Attaching subnets that are already registered with the
        Load Balancer has no effect.

        :type subnets: string or List of strings
        :param subnets: The name of the subnet(s) to add.

        """
    if isinstance(subnets, six.string_types):
        subnets = [subnets]
    new_subnets = self.connection.attach_lb_to_subnets(self.name, subnets)
    self.subnets = new_subnets