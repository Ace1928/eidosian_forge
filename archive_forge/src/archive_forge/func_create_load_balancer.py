from boto.connection import AWSQueryConnection
from boto.ec2.instanceinfo import InstanceInfo
from boto.ec2.elb.loadbalancer import LoadBalancer, LoadBalancerZones
from boto.ec2.elb.instancestate import InstanceState
from boto.ec2.elb.healthcheck import HealthCheck
from boto.regioninfo import RegionInfo, get_regions, load_regions
from boto.regioninfo import connect
import boto
from boto.compat import six
def create_load_balancer(self, name, zones, listeners=None, subnets=None, security_groups=None, scheme='internet-facing', complex_listeners=None):
    """
        Create a new load balancer for your account. By default the load
        balancer will be created in EC2. To create a load balancer inside a
        VPC, parameter zones must be set to None and subnets must not be None.
        The load balancer will be automatically created under the VPC that
        contains the subnet(s) specified.

        :type name: string
        :param name: The mnemonic name associated with the new load balancer

        :type zones: List of strings
        :param zones: The names of the availability zone(s) to add.

        :type listeners: List of tuples
        :param listeners: Each tuple contains three or four values,
            (LoadBalancerPortNumber, InstancePortNumber, Protocol,
            [SSLCertificateId]) where LoadBalancerPortNumber and
            InstancePortNumber are integer values between 1 and 65535,
            Protocol is a string containing either 'TCP', 'SSL', HTTP', or
            'HTTPS'; SSLCertificateID is the ARN of a AWS IAM
            certificate, and must be specified when doing HTTPS.

        :type subnets: list of strings
        :param subnets: A list of subnet IDs in your VPC to attach to
            your LoadBalancer.

        :type security_groups: list of strings
        :param security_groups: The security groups assigned to your
            LoadBalancer within your VPC.

        :type scheme: string
        :param scheme: The type of a LoadBalancer.  By default, Elastic
            Load Balancing creates an internet-facing LoadBalancer with
            a publicly resolvable DNS name, which resolves to public IP
            addresses.

            Specify the value internal for this option to create an
            internal LoadBalancer with a DNS name that resolves to
            private IP addresses.

            This option is only available for LoadBalancers attached
            to an Amazon VPC.

        :type complex_listeners: List of tuples
        :param complex_listeners: Each tuple contains four or five values,
            (LoadBalancerPortNumber, InstancePortNumber, Protocol,
             InstanceProtocol, SSLCertificateId).

            Where:
                - LoadBalancerPortNumber and InstancePortNumber are integer
                  values between 1 and 65535
                - Protocol and InstanceProtocol is a string containing
                  either 'TCP',
                  'SSL', 'HTTP', or 'HTTPS'
                - SSLCertificateId is the ARN of an SSL certificate loaded into
                  AWS IAM

        :rtype: :class:`boto.ec2.elb.loadbalancer.LoadBalancer`
        :return: The newly created
            :class:`boto.ec2.elb.loadbalancer.LoadBalancer`
        """
    if not listeners and (not complex_listeners):
        return None
    params = {'LoadBalancerName': name, 'Scheme': scheme}
    if listeners:
        for index, listener in enumerate(listeners):
            i = index + 1
            protocol = listener[2].upper()
            params['Listeners.member.%d.LoadBalancerPort' % i] = listener[0]
            params['Listeners.member.%d.InstancePort' % i] = listener[1]
            params['Listeners.member.%d.Protocol' % i] = listener[2]
            if protocol == 'HTTPS' or protocol == 'SSL':
                params['Listeners.member.%d.SSLCertificateId' % i] = listener[3]
    if complex_listeners:
        for index, listener in enumerate(complex_listeners):
            i = index + 1
            protocol = listener[2].upper()
            InstanceProtocol = listener[3].upper()
            params['Listeners.member.%d.LoadBalancerPort' % i] = listener[0]
            params['Listeners.member.%d.InstancePort' % i] = listener[1]
            params['Listeners.member.%d.Protocol' % i] = listener[2]
            params['Listeners.member.%d.InstanceProtocol' % i] = listener[3]
            if protocol == 'HTTPS' or protocol == 'SSL':
                params['Listeners.member.%d.SSLCertificateId' % i] = listener[4]
    if zones:
        self.build_list_params(params, zones, 'AvailabilityZones.member.%d')
    if subnets:
        self.build_list_params(params, subnets, 'Subnets.member.%d')
    if security_groups:
        self.build_list_params(params, security_groups, 'SecurityGroups.member.%d')
    load_balancer = self.get_object('CreateLoadBalancer', params, LoadBalancer)
    load_balancer.name = name
    load_balancer.listeners = listeners
    load_balancer.availability_zones = zones
    load_balancer.subnets = subnets
    load_balancer.security_groups = security_groups
    return load_balancer