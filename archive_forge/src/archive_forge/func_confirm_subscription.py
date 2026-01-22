import uuid
import hashlib
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.compat import json
import boto
def confirm_subscription(self, topic, token, authenticate_on_unsubscribe=False):
    """
        Get properties of a Topic

        :type topic: string
        :param topic: The ARN of the new topic.

        :type token: string
        :param token: Short-lived token sent to and endpoint during
                      the Subscribe operation.

        :type authenticate_on_unsubscribe: bool
        :param authenticate_on_unsubscribe: Optional parameter indicating
                                            that you wish to disable
                                            unauthenticated unsubscription
                                            of the subscription.

        """
    params = {'TopicArn': topic, 'Token': token}
    if authenticate_on_unsubscribe:
        params['AuthenticateOnUnsubscribe'] = 'true'
    return self._make_request('ConfirmSubscription', params)