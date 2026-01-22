import boto
from boto.dynamodb2 import exceptions
from boto.dynamodb2.fields import (HashKey, RangeKey,
from boto.dynamodb2.items import Item
from boto.dynamodb2.layer1 import DynamoDBConnection
from boto.dynamodb2.results import ResultSet, BatchGetResultSet
from boto.dynamodb2.types import (NonBooleanDynamizer, Dynamizer, FILTER_OPERATORS,
from boto.exception import JSONResponseError
def resend_unprocessed(self):
    boto.log.info('Re-sending %s unprocessed items.' % len(self._unprocessed))
    while len(self._unprocessed):
        to_resend = self._unprocessed[:25]
        self._unprocessed = self._unprocessed[25:]
        batch_data = {self.table.table_name: to_resend}
        boto.log.info('Sending %s items' % len(to_resend))
        resp = self.table.connection.batch_write_item(batch_data)
        self.handle_unprocessed(resp)
        boto.log.info('%s unprocessed items left' % len(self._unprocessed))