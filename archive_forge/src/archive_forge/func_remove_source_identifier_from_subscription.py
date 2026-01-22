import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.rds2 import exceptions
from boto.compat import json
def remove_source_identifier_from_subscription(self, subscription_name, source_identifier):
    """
        Removes a source identifier from an existing RDS event
        notification subscription.

        :type subscription_name: string
        :param subscription_name: The name of the RDS event notification
            subscription you want to remove a source identifier from.

        :type source_identifier: string
        :param source_identifier: The source identifier to be removed from the
            subscription, such as the **DB instance identifier** for a DB
            instance or the name of a security group.

        """
    params = {'SubscriptionName': subscription_name, 'SourceIdentifier': source_identifier}
    return self._make_request(action='RemoveSourceIdentifierFromSubscription', verb='POST', path='/', params=params)