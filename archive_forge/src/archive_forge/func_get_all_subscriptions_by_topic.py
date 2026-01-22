import uuid
import hashlib
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.compat import json
import boto
def get_all_subscriptions_by_topic(self, topic, next_token=None):
    """
        Get list of all subscriptions to a specific topic.

        :type topic: string
        :param topic: The ARN of the topic for which you wish to
                      find subscriptions.

        :type next_token: string
        :param next_token: Token returned by the previous call to
                           this method.

        """
    params = {'TopicArn': topic}
    if next_token:
        params['NextToken'] = next_token
    return self._make_request('ListSubscriptionsByTopic', params)