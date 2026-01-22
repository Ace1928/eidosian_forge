from boto.dynamodb.layer1 import Layer1
from boto.dynamodb.table import Table
from boto.dynamodb.schema import Schema
from boto.dynamodb.item import Item
from boto.dynamodb.batch import BatchList, BatchWriteList
from boto.dynamodb.types import get_dynamodb_type, Dynamizer, \
def next_response(self):
    """
        Issue a call and return the result.  You can invoke this method
        while iterating over the TableGenerator in order to skip to the
        next "page" of results.
        """
    limit = self.kwargs.get('limit')
    if self.remaining > 0 and (limit is None or limit > self.remaining):
        self.kwargs['limit'] = self.remaining
    self._response = self.callable(**self.kwargs)
    self.kwargs['limit'] = limit
    self._consumed_units += self._response.get('ConsumedCapacityUnits', 0.0)
    self._count += self._response.get('Count', 0)
    self._scanned_count += self._response.get('ScannedCount', 0)
    if 'LastEvaluatedKey' in self._response:
        lek = self._response['LastEvaluatedKey']
        esk = self.table.layer2.dynamize_last_evaluated_key(lek)
        self.kwargs['exclusive_start_key'] = esk
        lektuple = (lek['HashKeyElement'],)
        if 'RangeKeyElement' in lek:
            lektuple += (lek['RangeKeyElement'],)
        self.last_evaluated_key = lektuple
    else:
        self.last_evaluated_key = None
    return self._response