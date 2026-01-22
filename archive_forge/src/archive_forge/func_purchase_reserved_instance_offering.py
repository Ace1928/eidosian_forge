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
def purchase_reserved_instance_offering(self, reserved_instances_offering_id, instance_count=1, limit_price=None, dry_run=False):
    """
        Purchase a Reserved Instance for use with your account.
        ** CAUTION **
        This request can result in large amounts of money being charged to your
        AWS account.  Use with caution!

        :type reserved_instances_offering_id: string
        :param reserved_instances_offering_id: The offering ID of the Reserved
            Instance to purchase

        :type instance_count: int
        :param instance_count: The number of Reserved Instances to purchase.
            Default value is 1.

        :type limit_price: tuple
        :param instance_count: Limit the price on the total order.
            Must be a tuple of (amount, currency_code), for example:
            (100.0, 'USD').

        :type dry_run: bool
        :param dry_run: Set to True if the operation should not actually run.

        :rtype: :class:`boto.ec2.reservedinstance.ReservedInstance`
        :return: The newly created Reserved Instance
        """
    params = {'ReservedInstancesOfferingId': reserved_instances_offering_id, 'InstanceCount': instance_count}
    if limit_price is not None:
        params['LimitPrice.Amount'] = str(limit_price[0])
        params['LimitPrice.CurrencyCode'] = str(limit_price[1])
    if dry_run:
        params['DryRun'] = 'true'
    return self.get_object('PurchaseReservedInstancesOffering', params, ReservedInstance, verb='POST')