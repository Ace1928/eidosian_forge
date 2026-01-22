from boto.ec2.elb.healthcheck import HealthCheck
from boto.ec2.elb.listener import Listener
from boto.ec2.elb.listelement import ListElement
from boto.ec2.elb.policies import Policies, OtherPolicy
from boto.ec2.elb.securitygroup import SecurityGroup
from boto.ec2.instanceinfo import InstanceInfo
from boto.resultset import ResultSet
from boto.compat import six
def get_instance_health(self, instances=None):
    """
        Returns a list of :py:class:`boto.ec2.elb.instancestate.InstanceState`
        objects, which show the health of the instances attached to this
        load balancer.

        :rtype: list
        :returns: A list of
            :py:class:`InstanceState <boto.ec2.elb.instancestate.InstanceState>`
            instances, representing the instances
            attached to this load balancer.
        """
    return self.connection.describe_instance_health(self.name, instances)