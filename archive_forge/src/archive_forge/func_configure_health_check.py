from boto.ec2.elb.healthcheck import HealthCheck
from boto.ec2.elb.listener import Listener
from boto.ec2.elb.listelement import ListElement
from boto.ec2.elb.policies import Policies, OtherPolicy
from boto.ec2.elb.securitygroup import SecurityGroup
from boto.ec2.instanceinfo import InstanceInfo
from boto.resultset import ResultSet
from boto.compat import six
def configure_health_check(self, health_check):
    """
        Configures the health check behavior for the instances behind this
        load balancer. See :ref:`elb-configuring-a-health-check` for a
        walkthrough.

        :param boto.ec2.elb.healthcheck.HealthCheck health_check: A
            HealthCheck instance that tells the load balancer how to check
            its instances for health.
        """
    return self.connection.configure_health_check(self.name, health_check)