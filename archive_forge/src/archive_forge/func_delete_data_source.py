import boto
from boto.compat import json, urlsplit
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.machinelearning import exceptions
def delete_data_source(self, data_source_id):
    """
        Assigns the DELETED status to a `DataSource`, rendering it
        unusable.

        After using the `DeleteDataSource` operation, you can use the
        GetDataSource operation to verify that the status of the
        `DataSource` changed to DELETED.

        The results of the `DeleteDataSource` operation are
        irreversible.

        :type data_source_id: string
        :param data_source_id: A user-supplied ID that uniquely identifies the
            `DataSource`.

        """
    params = {'DataSourceId': data_source_id}
    return self.make_request(action='DeleteDataSource', body=json.dumps(params))