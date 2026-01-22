import os
from boto.compat import json
from boto.exception import JSONResponseError
from boto.connection import AWSAuthConnection
from boto.regioninfo import RegionInfo
from boto.awslambda import exceptions
def upload_function(self, function_name, function_zip, runtime, role, handler, mode, description=None, timeout=None, memory_size=None):
    """
        Creates a new Lambda function or updates an existing function.
        The function metadata is created from the request parameters,
        and the code for the function is provided by a .zip file in
        the request body. If the function name already exists, the
        existing Lambda function is updated with the new code and
        metadata.

        This operation requires permission for the
        `lambda:UploadFunction` action.

        :type function_name: string
        :param function_name: The name you want to assign to the function you
            are uploading. The function names appear in the console and are
            returned in the ListFunctions API. Function names are used to
            specify functions to other AWS Lambda APIs, such as InvokeAsync.

        :type function_zip: blob
        :param function_zip: A .zip file containing your packaged source code.
            For more information about creating a .zip file, go to `AWS LambdaL
            How it Works`_ in the AWS Lambda Developer Guide.

        :type runtime: string
        :param runtime: The runtime environment for the Lambda function you are
            uploading. Currently, Lambda supports only "nodejs" as the runtime.

        :type role: string
        :param role: The Amazon Resource Name (ARN) of the IAM role that Lambda
            assumes when it executes your function to access any other Amazon
            Web Services (AWS) resources.

        :type handler: string
        :param handler: The function that Lambda calls to begin execution. For
            Node.js, it is the module-name . export value in your function.

        :type mode: string
        :param mode: How the Lambda function will be invoked. Lambda supports
            only the "event" mode.

        :type description: string
        :param description: A short, user-defined function description. Lambda
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
            CPU and memory requirements. For example, database operation might
            need less memory compared to image processing function. The default
            value is 128 MB. The value must be a multiple of 64 MB.

        """
    uri = '/2014-11-13/functions/{0}'.format(function_name)
    headers = {}
    query_params = {}
    if runtime is not None:
        query_params['Runtime'] = runtime
    if role is not None:
        query_params['Role'] = role
    if handler is not None:
        query_params['Handler'] = handler
    if mode is not None:
        query_params['Mode'] = mode
    if description is not None:
        query_params['Description'] = description
    if timeout is not None:
        query_params['Timeout'] = timeout
    if memory_size is not None:
        query_params['MemorySize'] = memory_size
    try:
        content_length = str(len(function_zip))
    except (TypeError, AttributeError):
        try:
            function_zip.tell()
        except (AttributeError, OSError, IOError):
            raise TypeError('File-like object passed to parameter ``function_zip`` must be seekable.')
        content_length = str(os.fstat(function_zip.fileno()).st_size)
    headers['Content-Length'] = content_length
    return self.make_request('PUT', uri, expected_status=201, data=function_zip, headers=headers, params=query_params)