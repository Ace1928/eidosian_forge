import boto
from boto.dynamodb2 import exceptions
from boto.dynamodb2.fields import (HashKey, RangeKey,
from boto.dynamodb2.items import Item
from boto.dynamodb2.layer1 import DynamoDBConnection
from boto.dynamodb2.results import ResultSet, BatchGetResultSet
from boto.dynamodb2.types import (NonBooleanDynamizer, Dynamizer, FILTER_OPERATORS,
from boto.exception import JSONResponseError
def batch_get(self, keys, consistent=False, attributes=None):
    """
        Fetches many specific items in batch from a table.

        Requires a ``keys`` parameter, which should be a list of dictionaries.
        Each dictionary should consist of the keys values to specify.

        Optionally accepts a ``consistent`` parameter, which should be a
        boolean. If you provide ``True``, a strongly consistent read will be
        used. (Default: False)

        Optionally accepts an ``attributes`` parameter, which should be a
        tuple. If you provide any attributes only these will be fetched
        from DynamoDB.

        Returns a ``ResultSet``, which transparently handles the pagination of
        results you get back.

        Example::

            >>> results = users.batch_get(keys=[
            ...     {
            ...         'username': 'johndoe',
            ...     },
            ...     {
            ...         'username': 'jane',
            ...     },
            ...     {
            ...         'username': 'fred',
            ...     },
            ... ])
            >>> for res in results:
            ...     print res['first_name']
            'John'
            'Jane'
            'Fred'

        """
    results = BatchGetResultSet(keys=keys, max_batch_get=self.max_batch_get)
    results.to_call(self._batch_get, consistent=consistent, attributes=attributes)
    return results