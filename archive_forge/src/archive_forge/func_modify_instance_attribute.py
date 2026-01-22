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
def modify_instance_attribute(self, instance_id, attribute, value, dry_run=False):
    """
        Changes an attribute of an instance

        :type instance_id: string
        :param instance_id: The instance id you wish to change

        :type attribute: string
        :param attribute: The attribute you wish to change.

            * instanceType - A valid instance type (m1.small)
            * kernel - Kernel ID (None)
            * ramdisk - Ramdisk ID (None)
            * userData - Base64 encoded String (None)
            * disableApiTermination - Boolean (true)
            * instanceInitiatedShutdownBehavior - stop|terminate
            * blockDeviceMapping - List of strings - ie: ['/dev/sda=false']
            * sourceDestCheck - Boolean (true)
            * groupSet - Set of Security Groups or IDs
            * ebsOptimized - Boolean (false)
            * sriovNetSupport - String - ie: 'simple'

        :type value: string
        :param value: The new value for the attribute

        :type dry_run: bool
        :param dry_run: Set to True if the operation should not actually run.

        :rtype: bool
        :return: Whether the operation succeeded or not
        """
    bool_reqs = ('disableapitermination', 'sourcedestcheck', 'ebsoptimized')
    if attribute.lower() in bool_reqs:
        if isinstance(value, bool):
            if value:
                value = 'true'
            else:
                value = 'false'
    params = {'InstanceId': instance_id}
    if attribute.lower() == 'groupset':
        for idx, sg in enumerate(value):
            if isinstance(sg, SecurityGroup):
                sg = sg.id
            params['GroupId.%s' % (idx + 1)] = sg
    elif attribute.lower() == 'blockdevicemapping':
        for idx, kv in enumerate(value):
            dev_name, _, flag = kv.partition('=')
            pre = 'BlockDeviceMapping.%d' % (idx + 1)
            params['%s.DeviceName' % pre] = dev_name
            params['%s.Ebs.DeleteOnTermination' % pre] = flag or 'true'
    else:
        attribute = attribute[0].upper() + attribute[1:]
        params['%s.Value' % attribute] = value
    if dry_run:
        params['DryRun'] = 'true'
    return self.get_status('ModifyInstanceAttribute', params, verb='POST')