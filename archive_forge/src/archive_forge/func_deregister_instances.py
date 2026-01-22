from boto.ec2.elb.healthcheck import HealthCheck
from boto.ec2.elb.listener import Listener
from boto.ec2.elb.listelement import ListElement
from boto.ec2.elb.policies import Policies, OtherPolicy
from boto.ec2.elb.securitygroup import SecurityGroup
from boto.ec2.instanceinfo import InstanceInfo
from boto.resultset import ResultSet
from boto.compat import six
def deregister_instances(self, instances):
    """
        Remove instances from this load balancer. Removing instances that are
        not registered with the load balancer has no effect.

        :param list instances: List of instance IDs (strings) that you'd like
            to remove from this load balancer.

        """
    if isinstance(instances, six.string_types):
        instances = [instances]
    new_instances = self.connection.deregister_instances(self.name, instances)
    self.instances = new_instances