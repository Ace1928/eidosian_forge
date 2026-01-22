import boto
from boto.dynamodb2 import exceptions
from boto.dynamodb2.fields import (HashKey, RangeKey,
from boto.dynamodb2.items import Item
from boto.dynamodb2.layer1 import DynamoDBConnection
from boto.dynamodb2.results import ResultSet, BatchGetResultSet
from boto.dynamodb2.types import (NonBooleanDynamizer, Dynamizer, FILTER_OPERATORS,
from boto.exception import JSONResponseError
def delete_global_secondary_index(self, global_index_name):
    """
        Deletes a global index in DynamoDB after the table has been created.

        Requires a ``global_index_name`` parameter, which should be a simple
        string of the name of the global secondary index.

        To update ``global_indexes`` information on the ``Table``, you'll need
        to call ``Table.describe``.

        Returns ``True`` on success.

        Example::

            # To delete a global index
            >>> users.delete_global_secondary_index('TheIndexNameHere')
            True

        """
    if global_index_name:
        gsi_data = [{'Delete': {'IndexName': global_index_name}}]
        self.connection.update_table(self.table_name, global_secondary_index_updates=gsi_data)
        return True
    else:
        msg = 'You need to provide the global index name to delete_global_secondary_index method'
        boto.log.error(msg)
        return False