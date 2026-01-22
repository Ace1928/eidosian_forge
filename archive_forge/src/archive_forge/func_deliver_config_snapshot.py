import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.configservice import exceptions
def deliver_config_snapshot(self, delivery_channel_name):
    """
        Schedules delivery of a configuration snapshot to the Amazon
        S3 bucket in the specified delivery channel. After the
        delivery has started, AWS Config sends following notifications
        using an Amazon SNS topic that you have specified.


        + Notification of starting the delivery.
        + Notification of delivery completed, if the delivery was
          successfully completed.
        + Notification of delivery failure, if the delivery failed to
          complete.

        :type delivery_channel_name: string
        :param delivery_channel_name: The name of the delivery channel through
            which the snapshot is delivered.

        """
    params = {'deliveryChannelName': delivery_channel_name}
    return self.make_request(action='DeliverConfigSnapshot', body=json.dumps(params))