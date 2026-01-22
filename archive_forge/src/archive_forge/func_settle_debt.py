import urllib
import uuid
from boto.connection import AWSQueryConnection
from boto.fps.exception import ResponseErrorFactory
from boto.fps.response import ResponseFactory
import boto.fps.response
@needs_caller_reference
@complex_amounts('SettlementAmount')
@requires(['CreditInstrumentId', 'SettlementAmount.Value', 'SenderTokenId', 'SettlementAmount.CurrencyCode'])
@api_action()
def settle_debt(self, action, response, **kw):
    """
        Allows a caller to initiate a transaction that atomically transfers
        money from a sender's payment instrument to the recipient, while
        decreasing corresponding debt balance.
        """
    return self.get_object(action, kw, response)