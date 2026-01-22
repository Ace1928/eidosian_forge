import os
from boto.compat import json
from boto.exception import JSONResponseError
from boto.connection import AWSAuthConnection
from boto.regioninfo import RegionInfo
from boto.awslambda import exceptions
def get_function_configuration(self, function_name):
    """
        Returns the configuration information of the Lambda function.
        This the same information you provided as parameters when
        uploading the function by using UploadFunction.

        This operation requires permission for the
        `lambda:GetFunctionConfiguration` operation.

        :type function_name: string
        :param function_name: The name of the Lambda function for which you
            want to retrieve the configuration information.

        """
    uri = '/2014-11-13/functions/{0}/configuration'.format(function_name)
    return self.make_request('GET', uri, expected_status=200)