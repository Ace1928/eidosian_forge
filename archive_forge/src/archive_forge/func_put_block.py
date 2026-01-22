import logging
import urllib
from copy import deepcopy
from mlflow.utils import rest_utils
from mlflow.utils.file_utils import read_chunk
def put_block(sas_url, block_id, data, headers):
    """
    Performs an Azure `Put Block` operation
    (https://docs.microsoft.com/en-us/rest/api/storageservices/put-block)

    Args:
        sas_url: A shared access signature URL referring to the Azure Block Blob
            to which the specified data should be staged.
        block_id: A base64-encoded string identifying the block.
        data: Data to include in the Put Block request body.
        headers: Additional headers to include in the Put Block request body
            (the `x-ms-blob-type` header is always included automatically).
    """
    request_url = _append_query_parameters(sas_url, {'comp': 'block', 'blockid': block_id})
    request_headers = deepcopy(_PUT_BLOCK_HEADERS)
    for name, value in headers.items():
        if _is_valid_put_block_header(name):
            request_headers[name] = value
        else:
            _logger.debug("Removed unsupported '%s' header for Put Block operation", name)
    with rest_utils.cloud_storage_http_request('put', request_url, data=data, headers=request_headers) as response:
        rest_utils.augmented_raise_for_status(response)