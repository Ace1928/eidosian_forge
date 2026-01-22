import urllib
import uuid
from boto.connection import AWSQueryConnection
from boto.fps.exception import ResponseErrorFactory
from boto.fps.response import ResponseFactory
import boto.fps.response
@needs_caller_reference
@complex_amounts('RefundAmount')
@requires(['SubscriptionId'])
@api_action()
def cancel_subscription_and_refund(self, action, response, **kw):
    """
        Cancels a subscription.
        """
    message = 'If you specify a RefundAmount, you must specify CallerReference.'
    assert not 'RefundAmount.Value' in kw or 'CallerReference' in kw, message
    return self.get_object(action, kw, response)