import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.route53.domains import exceptions
def get_operation_detail(self, operation_id):
    """
        This operation returns the current status of an operation that
        is not completed.

        :type operation_id: string
        :param operation_id: The identifier for the operation for which you
            want to get the status. Amazon Route 53 returned the identifier in
            the response to the original request.
        Type: String

        Default: None

        Required: Yes

        """
    params = {'OperationId': operation_id}
    return self.make_request(action='GetOperationDetail', body=json.dumps(params))