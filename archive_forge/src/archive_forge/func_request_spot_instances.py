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
def request_spot_instances(self, price, image_id, count=1, type='one-time', valid_from=None, valid_until=None, launch_group=None, availability_zone_group=None, key_name=None, security_groups=None, user_data=None, addressing_type=None, instance_type='m1.small', placement=None, kernel_id=None, ramdisk_id=None, monitoring_enabled=False, subnet_id=None, placement_group=None, block_device_map=None, instance_profile_arn=None, instance_profile_name=None, security_group_ids=None, ebs_optimized=False, network_interfaces=None, dry_run=False):
    """
        Request instances on the spot market at a particular price.

        :type price: str
        :param price: The maximum price of your bid

        :type image_id: string
        :param image_id: The ID of the image to run

        :type count: int
        :param count: The of instances to requested

        :type type: str
        :param type: Type of request. Can be 'one-time' or 'persistent'.
                     Default is one-time.

        :type valid_from: str
        :param valid_from: Start date of the request. An ISO8601 time string.

        :type valid_until: str
        :param valid_until: End date of the request.  An ISO8601 time string.

        :type launch_group: str
        :param launch_group: If supplied, all requests will be fulfilled
            as a group.

        :type availability_zone_group: str
        :param availability_zone_group: If supplied, all requests will be
            fulfilled within a single availability zone.

        :type key_name: string
        :param key_name: The name of the key pair with which to
            launch instances

        :type security_groups: list of strings
        :param security_groups: The names of the security groups with which to
            associate instances

        :type user_data: string
        :param user_data: The user data passed to the launched instances

        :type instance_type: string
        :param instance_type: The type of instance to run:

            * t1.micro
            * m1.small
            * m1.medium
            * m1.large
            * m1.xlarge
            * m3.medium
            * m3.large
            * m3.xlarge
            * m3.2xlarge
            * c1.medium
            * c1.xlarge
            * m2.xlarge
            * m2.2xlarge
            * m2.4xlarge
            * cr1.8xlarge
            * hi1.4xlarge
            * hs1.8xlarge
            * cc1.4xlarge
            * cg1.4xlarge
            * cc2.8xlarge
            * g2.2xlarge
            * c3.large
            * c3.xlarge
            * c3.2xlarge
            * c3.4xlarge
            * c3.8xlarge
            * c4.large
            * c4.xlarge
            * c4.2xlarge
            * c4.4xlarge
            * c4.8xlarge
            * i2.xlarge
            * i2.2xlarge
            * i2.4xlarge
            * i2.8xlarge
            * t2.micro
            * t2.small
            * t2.medium

        :type placement: string
        :param placement: The availability zone in which to launch
            the instances

        :type kernel_id: string
        :param kernel_id: The ID of the kernel with which to launch the
            instances

        :type ramdisk_id: string
        :param ramdisk_id: The ID of the RAM disk with which to launch the
            instances

        :type monitoring_enabled: bool
        :param monitoring_enabled: Enable detailed CloudWatch monitoring on
            the instance.

        :type subnet_id: string
        :param subnet_id: The subnet ID within which to launch the instances
            for VPC.

        :type placement_group: string
        :param placement_group: If specified, this is the name of the placement
            group in which the instance(s) will be launched.

        :type block_device_map: :class:`boto.ec2.blockdevicemapping.BlockDeviceMapping`
        :param block_device_map: A BlockDeviceMapping data structure
            describing the EBS volumes associated with the Image.

        :type security_group_ids: list of strings
        :param security_group_ids: The ID of the VPC security groups with
            which to associate instances.

        :type instance_profile_arn: string
        :param instance_profile_arn: The Amazon resource name (ARN) of
            the IAM Instance Profile (IIP) to associate with the instances.

        :type instance_profile_name: string
        :param instance_profile_name: The name of
            the IAM Instance Profile (IIP) to associate with the instances.

        :type ebs_optimized: bool
        :param ebs_optimized: Whether the instance is optimized for
            EBS I/O.  This optimization provides dedicated throughput
            to Amazon EBS and an optimized configuration stack to
            provide optimal EBS I/O performance.  This optimization
            isn't available with all instance types.

        :type network_interfaces: list
        :param network_interfaces: A list of
            :class:`boto.ec2.networkinterface.NetworkInterfaceSpecification`

        :type dry_run: bool
        :param dry_run: Set to True if the operation should not actually run.

        :rtype: Reservation
        :return: The :class:`boto.ec2.spotinstancerequest.SpotInstanceRequest`
                 associated with the request for machines
        """
    ls = 'LaunchSpecification'
    params = {'%s.ImageId' % ls: image_id, 'Type': type, 'SpotPrice': price}
    if count:
        params['InstanceCount'] = count
    if valid_from:
        params['ValidFrom'] = valid_from
    if valid_until:
        params['ValidUntil'] = valid_until
    if launch_group:
        params['LaunchGroup'] = launch_group
    if availability_zone_group:
        params['AvailabilityZoneGroup'] = availability_zone_group
    if key_name:
        params['%s.KeyName' % ls] = key_name
    if security_group_ids:
        l = []
        for group in security_group_ids:
            if isinstance(group, SecurityGroup):
                l.append(group.id)
            else:
                l.append(group)
        self.build_list_params(params, l, '%s.SecurityGroupId' % ls)
    if security_groups:
        l = []
        for group in security_groups:
            if isinstance(group, SecurityGroup):
                l.append(group.name)
            else:
                l.append(group)
        self.build_list_params(params, l, '%s.SecurityGroup' % ls)
    if user_data:
        params['%s.UserData' % ls] = base64.b64encode(user_data)
    if addressing_type:
        params['%s.AddressingType' % ls] = addressing_type
    if instance_type:
        params['%s.InstanceType' % ls] = instance_type
    if placement:
        params['%s.Placement.AvailabilityZone' % ls] = placement
    if kernel_id:
        params['%s.KernelId' % ls] = kernel_id
    if ramdisk_id:
        params['%s.RamdiskId' % ls] = ramdisk_id
    if monitoring_enabled:
        params['%s.Monitoring.Enabled' % ls] = 'true'
    if subnet_id:
        params['%s.SubnetId' % ls] = subnet_id
    if placement_group:
        params['%s.Placement.GroupName' % ls] = placement_group
    if block_device_map:
        block_device_map.ec2_build_list_params(params, '%s.' % ls)
    if instance_profile_name:
        params['%s.IamInstanceProfile.Name' % ls] = instance_profile_name
    if instance_profile_arn:
        params['%s.IamInstanceProfile.Arn' % ls] = instance_profile_arn
    if ebs_optimized:
        params['%s.EbsOptimized' % ls] = 'true'
    if network_interfaces:
        network_interfaces.build_list_params(params, prefix=ls + '.')
    if dry_run:
        params['DryRun'] = 'true'
    return self.get_list('RequestSpotInstances', params, [('item', SpotInstanceRequest)], verb='POST')