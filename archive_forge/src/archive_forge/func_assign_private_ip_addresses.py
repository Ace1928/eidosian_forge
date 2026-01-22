import base64
import warnings
from datetime import datetime
from datetime import timedelta
import boto
from boto.auth import detect_potential_sigv4
from boto.connection import AWSQueryConnection
from boto.resultset import ResultSet
from boto.ec2.image import Image, ImageAttribute, CopyImage
from boto.ec2.instance import Reservation, Instance
from boto.ec2.instance import ConsoleOutput, InstanceAttribute
from boto.ec2.keypair import KeyPair
from boto.ec2.address import Address
from boto.ec2.volume import Volume, VolumeAttribute
from boto.ec2.snapshot import Snapshot
from boto.ec2.snapshot import SnapshotAttribute
from boto.ec2.zone import Zone
from boto.ec2.securitygroup import SecurityGroup
from boto.ec2.regioninfo import RegionInfo
from boto.ec2.instanceinfo import InstanceInfo
from boto.ec2.reservedinstance import ReservedInstancesOffering
from boto.ec2.reservedinstance import ReservedInstance
from boto.ec2.reservedinstance import ReservedInstanceListing
from boto.ec2.reservedinstance import ReservedInstancesConfiguration
from boto.ec2.reservedinstance import ModifyReservedInstancesResult
from boto.ec2.reservedinstance import ReservedInstancesModification
from boto.ec2.spotinstancerequest import SpotInstanceRequest
from boto.ec2.spotpricehistory import SpotPriceHistory
from boto.ec2.spotdatafeedsubscription import SpotDatafeedSubscription
from boto.ec2.bundleinstance import BundleInstanceTask
from boto.ec2.placementgroup import PlacementGroup
from boto.ec2.tag import Tag
from boto.ec2.instancetype import InstanceType
from boto.ec2.instancestatus import InstanceStatusSet
from boto.ec2.volumestatus import VolumeStatusSet
from boto.ec2.networkinterface import NetworkInterface
from boto.ec2.attributes import AccountAttribute, VPCAttribute
from boto.ec2.blockdevicemapping import BlockDeviceMapping, BlockDeviceType
from boto.exception import EC2ResponseError
from boto.compat import six
def assign_private_ip_addresses(self, network_interface_id=None, private_ip_addresses=None, secondary_private_ip_address_count=None, allow_reassignment=False, dry_run=False):
    """
        Assigns one or more secondary private IP addresses to a network
        interface in Amazon VPC.

        :type network_interface_id: string
        :param network_interface_id: The network interface to which the IP
            address will be assigned.

        :type private_ip_addresses: list
        :param private_ip_addresses: Assigns the specified IP addresses as
            secondary IP addresses to the network interface.

        :type secondary_private_ip_address_count: int
        :param secondary_private_ip_address_count: The number of secondary IP
            addresses to assign to the network interface. You cannot specify
            this parameter when also specifying private_ip_addresses.

        :type allow_reassignment: bool
        :param allow_reassignment: Specifies whether to allow an IP address
            that is already assigned to another network interface or instance
            to be reassigned to the specified network interface.

        :type dry_run: bool
        :param dry_run: Set to True if the operation should not actually run.

        :rtype: bool
        :return: True if successful
        """
    params = {}
    if network_interface_id is not None:
        params['NetworkInterfaceId'] = network_interface_id
    if private_ip_addresses is not None:
        self.build_list_params(params, private_ip_addresses, 'PrivateIpAddress')
    elif secondary_private_ip_address_count is not None:
        params['SecondaryPrivateIpAddressCount'] = secondary_private_ip_address_count
    if allow_reassignment:
        params['AllowReassignment'] = 'true'
    if dry_run:
        params['DryRun'] = 'true'
    return self.get_status('AssignPrivateIpAddresses', params, verb='POST')