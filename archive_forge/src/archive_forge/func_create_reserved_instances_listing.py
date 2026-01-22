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
def create_reserved_instances_listing(self, reserved_instances_id, instance_count, price_schedules, client_token, dry_run=False):
    """Creates a new listing for Reserved Instances.

        Creates a new listing for Amazon EC2 Reserved Instances that will be
        sold in the Reserved Instance Marketplace. You can submit one Reserved
        Instance listing at a time.

        The Reserved Instance Marketplace matches sellers who want to resell
        Reserved Instance capacity that they no longer need with buyers who
        want to purchase additional capacity. Reserved Instances bought and
        sold through the Reserved Instance Marketplace work like any other
        Reserved Instances.

        If you want to sell your Reserved Instances, you must first register as
        a Seller in the Reserved Instance Marketplace. After completing the
        registration process, you can create a Reserved Instance Marketplace
        listing of some or all of your Reserved Instances, and specify the
        upfront price you want to receive for them. Your Reserved Instance
        listings then become available for purchase.

        :type reserved_instances_id: string
        :param reserved_instances_id: The ID of the Reserved Instance that
            will be listed.

        :type instance_count: int
        :param instance_count: The number of instances that are a part of a
            Reserved Instance account that will be listed in the Reserved
            Instance Marketplace. This number should be less than or equal to
            the instance count associated with the Reserved Instance ID
            specified in this call.

        :type price_schedules: List of tuples
        :param price_schedules: A list specifying the price of the Reserved
            Instance for each month remaining in the Reserved Instance term.
            Each tuple contains two elements, the price and the term.  For
            example, for an instance that 11 months remaining in its term,
            we can have a price schedule with an upfront price of $2.50.
            At 8 months remaining we can drop the price down to $2.00.
            This would be expressed as::

                price_schedules=[('2.50', 11), ('2.00', 8)]

        :type client_token: string
        :param client_token: Unique, case-sensitive identifier you provide
            to ensure idempotency of the request.  Maximum 64 ASCII characters.

        :type dry_run: bool
        :param dry_run: Set to True if the operation should not actually run.

        :rtype: list
        :return: A list of
            :class:`boto.ec2.reservedinstance.ReservedInstanceListing`

        """
    params = {'ReservedInstancesId': reserved_instances_id, 'InstanceCount': str(instance_count), 'ClientToken': client_token}
    for i, schedule in enumerate(price_schedules):
        price, term = schedule
        params['PriceSchedules.%s.Price' % i] = str(price)
        params['PriceSchedules.%s.Term' % i] = str(term)
    if dry_run:
        params['DryRun'] = 'true'
    return self.get_list('CreateReservedInstancesListing', params, [('item', ReservedInstanceListing)], verb='POST')