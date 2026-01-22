import os
from boto.compat import json
from boto.exception import JSONResponseError
from boto.connection import AWSAuthConnection
from boto.regioninfo import RegionInfo
from boto.awslambda import exceptions
def update_function_configuration(self, function_name, role=None, handler=None, description=None, timeout=None, memory_size=None):
    """
        Updates the configuration parameters for the specified Lambda
        function by using the values provided in the request. You
        provide only the parameters you want to change. This operation
        must only be used on an existing Lambda function and cannot be
        used to update the function's code.

        This operation requires permission for the
        `lambda:UpdateFunctionConfiguration` action.

        :type function_name: string
        :param function_name: The name of the Lambda function.

        :type role: string
        :param role: The Amazon Resource Name (ARN) of the IAM role that Lambda
            will assume when it executes your function.

        :type handler: string
        :param handler: The function that Lambda calls to begin executing your
            function. For Node.js, it is the module-name.export value in your
            function.

        :type description: string
        :param description: A short user-defined function description. Lambda
            does not use this value. Assign a meaningful description as you see
            fit.

        :type timeout: integer
        :param timeout: The function execution time at which Lambda should
            terminate the function. Because the execution time has cost
            implications, we recommend you set this value based on your
            expected execution time. The default is 3 seconds.

        :type memory_size: integer
        :param memory_size: The amount of memory, in MB, your Lambda function
            is given. Lambda uses this memory size to infer the amount of CPU
            allocated to your function. Your function use-case determines your
            CPU and memory requirements. For example, a database operation
            might need less memory compared to an image processing function.
            The default value is 128 MB. The value must be a multiple of 64 MB.

        """
    uri = '/2014-11-13/functions/{0}/configuration'.format(function_name)
    params = {}
    headers = {}
    query_params = {}
    if role is not None:
        query_params['Role'] = role
    if handler is not None:
        query_params['Handler'] = handler
    if description is not None:
        query_params['Description'] = description
    if timeout is not None:
        query_params['Timeout'] = timeout
    if memory_size is not None:
        query_params['MemorySize'] = memory_size
    return self.make_request('PUT', uri, expected_status=200, data=json.dumps(params), headers=headers, params=query_params)