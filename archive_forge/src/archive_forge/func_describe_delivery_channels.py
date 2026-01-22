import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.configservice import exceptions
def describe_delivery_channels(self, delivery_channel_names=None):
    """
        Returns details about the specified delivery channel. If a
        delivery channel is not specified, this action returns the
        details of all delivery channels associated with the account.

        :type delivery_channel_names: list
        :param delivery_channel_names: A list of delivery channel names.

        """
    params = {}
    if delivery_channel_names is not None:
        params['DeliveryChannelNames'] = delivery_channel_names
    return self.make_request(action='DescribeDeliveryChannels', body=json.dumps(params))