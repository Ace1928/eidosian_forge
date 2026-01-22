import urllib
import uuid
from boto.connection import AWSQueryConnection
from boto.fps.exception import ResponseErrorFactory
from boto.fps.response import ResponseFactory
import boto.fps.response
@needs_caller_reference
@complex_amounts('AdjustmentAmount')
@requires(['CreditInstrumentId', 'AdjustmentAmount.Value', 'AdjustmentAmount.CurrencyCode'])
@api_action()
def write_off_debt(self, action, response, **kw):
    """
        Allows a creditor to write off the debt balance accumulated partially
        or fully at any time.
        """
    return self.get_object(action, kw, response)