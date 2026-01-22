from libcloud.utils.xml import findall, findtext
from libcloud.common.aws import AWSGenericResponse, SignedAWSConnection
from libcloud.loadbalancer.base import Driver, Member, LoadBalancer
from libcloud.loadbalancer.types import State
def ex_create_target_group(self, name, port, proto, vpc, health_check_interval=30, health_check_path='/', health_check_port='traffic-port', health_check_proto='HTTP', health_check_timeout=5, health_check_matcher='200', healthy_threshold=5, unhealthy_threshold=2):
    """
        Create a target group for AWS ALB load balancer.

        :param name: Name of target group
        :type name: ``str``

        :param port: The port on which the targets receive traffic.
                    This port is used unless you specify a port override when
                    registering the target.
        :type port: ``int``

        :param proto: The protocol to use for routing traffic to the targets.
                    Can be 'HTTP' or 'HTTPS'.
        :type proto: ``str``

        :param vpc: The identifier of the virtual private cloud (VPC).
        :type vpc: ``str``

        :param health_check_interval: The approximate amount of time, in
                                    seconds, between health checks of an
                                    individual target. The default is
                                    30 seconds.
        :type health_check_interval: ``int``

        :param health_check_path: The ping path that is the destination on
                                the targets for health checks. The default is /
        :type health_check_path: ``str``

        :param health_check_port: The port the load balancer uses when
                                performing health checks on targets.
                                The default is traffic-port, which indicates
                                the port on which each target receives traffic
                                from the load balancer.
        :type health_check_port: ``str``

        :param health_check_proto: The protocol the load balancer uses when
                                performing health checks on targets.
                                Can be 'HTTP' (default) or 'HTTPS'.
        :type health_check_proto: ``str``

        :param health_check_timeout: The amount of time, in seconds, during
                                    which no response from a target means
                                    a failed health check. The default is 5s.
        :type health_check_timeout: ``int``

        :param health_check_matcher: The HTTP codes to use when checking for
                                    a successful response from a target.
                                    Valid values: "200", "200,202", "200-299".
        :type health_check_matcher: ``str``

        :param healthy_threshold: The number of consecutive health checks
                                  successes required before considering
                                  an unhealthy target healthy. The default is 5
        :type healthy_threshold: ``int``

        :param unhealthy_threshold: The number of consecutive health check
                                    failures required before considering
                                    a target unhealthy. The default is 2.
        :type unhealthy_threshold: ``int``

        :return: Target group object.
        :rtype: :class:`ALBTargetGroup`
        """
    params = {'Action': 'CreateTargetGroup', 'Name': name, 'Protocol': proto, 'Port': port, 'VpcId': vpc}
    params.update({'HealthCheckIntervalSeconds': health_check_interval, 'HealthCheckPath': health_check_path, 'HealthCheckPort': health_check_port, 'HealthCheckProtocol': health_check_proto, 'HealthCheckTimeoutSeconds': health_check_timeout, 'HealthyThresholdCount': healthy_threshold, 'UnhealthyThresholdCount': unhealthy_threshold, 'Matcher.HttpCode': health_check_matcher})
    data = self.connection.request(ROOT, params=params).object
    xpath = 'CreateTargetGroupResult/TargetGroups/member'
    for el in findall(element=data, xpath=xpath, namespace=NS):
        target_group = self._to_target_group(el)
    return target_group