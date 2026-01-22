from boto.ec2.connection import EC2Connection
from boto.resultset import ResultSet
from boto.vpc.vpc import VPC
from boto.vpc.customergateway import CustomerGateway
from boto.vpc.networkacl import NetworkAcl
from boto.vpc.routetable import RouteTable
from boto.vpc.internetgateway import InternetGateway
from boto.vpc.vpngateway import VpnGateway, Attachment
from boto.vpc.dhcpoptions import DhcpOptions
from boto.vpc.subnet import Subnet
from boto.vpc.vpnconnection import VpnConnection
from boto.vpc.vpc_peering_connection import VpcPeeringConnection
from boto.ec2 import RegionData
from boto.regioninfo import RegionInfo, get_regions
from boto.regioninfo import connect
def create_dhcp_options(self, domain_name=None, domain_name_servers=None, ntp_servers=None, netbios_name_servers=None, netbios_node_type=None, dry_run=False):
    """
        Create a new DhcpOption

        This corresponds to
        http://docs.amazonwebservices.com/AWSEC2/latest/APIReference/ApiReference-query-CreateDhcpOptions.html

        :type domain_name: str
        :param domain_name: A domain name of your choice (for example,
            example.com)

        :type domain_name_servers: list of strings
        :param domain_name_servers: The IP address of a domain name server. You
            can specify up to four addresses.

        :type ntp_servers: list of strings
        :param ntp_servers: The IP address of a Network Time Protocol (NTP)
            server. You can specify up to four addresses.

        :type netbios_name_servers: list of strings
        :param netbios_name_servers: The IP address of a NetBIOS name server.
            You can specify up to four addresses.

        :type netbios_node_type: str
        :param netbios_node_type: The NetBIOS node type (1, 2, 4, or 8). For
            more information about the values, see RFC 2132. We recommend you
            only use 2 at this time (broadcast and multicast are currently not
            supported).

        :type dry_run: bool
        :param dry_run: Set to True if the operation should not actually run.

        :rtype: The newly created DhcpOption
        :return: A :class:`boto.vpc.customergateway.DhcpOption` object
        """
    key_counter = 1
    params = {}

    def insert_option(params, name, value):
        params['DhcpConfiguration.%d.Key' % (key_counter,)] = name
        if isinstance(value, (list, tuple)):
            for idx, value in enumerate(value, 1):
                key_name = 'DhcpConfiguration.%d.Value.%d' % (key_counter, idx)
                params[key_name] = value
        else:
            key_name = 'DhcpConfiguration.%d.Value.1' % (key_counter,)
            params[key_name] = value
        return key_counter + 1
    if domain_name:
        key_counter = insert_option(params, 'domain-name', domain_name)
    if domain_name_servers:
        key_counter = insert_option(params, 'domain-name-servers', domain_name_servers)
    if ntp_servers:
        key_counter = insert_option(params, 'ntp-servers', ntp_servers)
    if netbios_name_servers:
        key_counter = insert_option(params, 'netbios-name-servers', netbios_name_servers)
    if netbios_node_type:
        key_counter = insert_option(params, 'netbios-node-type', netbios_node_type)
    if dry_run:
        params['DryRun'] = 'true'
    return self.get_object('CreateDhcpOptions', params, DhcpOptions)