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
def get_all_instance_status(self, instance_ids=None, max_results=None, next_token=None, filters=None, dry_run=False, include_all_instances=False):
    """
        Retrieve all the instances in your account scheduled for maintenance.

        :type instance_ids: list
        :param instance_ids: A list of strings of instance IDs

        :type max_results: int
        :param max_results: The maximum number of paginated instance
            items per response.

        :type next_token: str
        :param next_token: A string specifying the next paginated set
            of results to return.

        :type filters: dict
        :param filters: Optional filters that can be used to limit
            the results returned.  Filters are provided
            in the form of a dictionary consisting of
            filter names as the key and filter values
            as the value.  The set of allowable filter
            names/values is dependent on the request
            being performed.  Check the EC2 API guide
            for details.

        :type dry_run: bool
        :param dry_run: Set to True if the operation should not actually run.

        :type include_all_instances: bool
        :param include_all_instances: Set to True if all
            instances should be returned. (Only running
            instances are included by default.)

        :rtype: list
        :return: A list of instances that have maintenance scheduled.
        """
    params = {}
    if instance_ids:
        self.build_list_params(params, instance_ids, 'InstanceId')
    if max_results:
        params['MaxResults'] = max_results
    if next_token:
        params['NextToken'] = next_token
    if filters:
        self.build_filter_params(params, filters)
    if dry_run:
        params['DryRun'] = 'true'
    if include_all_instances:
        params['IncludeAllInstances'] = 'true'
    return self.get_object('DescribeInstanceStatus', params, InstanceStatusSet, verb='POST')