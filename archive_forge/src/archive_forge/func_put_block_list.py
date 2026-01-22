import logging
import urllib
from copy import deepcopy
from mlflow.utils import rest_utils
from mlflow.utils.file_utils import read_chunk
def put_block_list(sas_url, block_list, headers):
    """Performs an Azure `Put Block List` operation
    (https://docs.microsoft.com/en-us/rest/api/storageservices/put-block-list)

    Args:
        sas_url: A shared access signature URL referring to the Azure Block Blob
            to which the specified data should be staged.
        block_list: A list of uncommitted base64-encoded string block IDs to commit. For
            more information, see
            https://docs.microsoft.com/en-us/rest/api/storageservices/put-block-list.
        headers: Headers to include in the Put Block request body.

    """
    request_url = _append_query_parameters(sas_url, {'comp': 'blocklist'})
    data = _build_block_list_xml(block_list)
    request_headers = {}
    for name, value in headers.items():
        if _is_valid_put_block_list_header(name):
            request_headers[name] = value
        else:
            _logger.debug("Removed unsupported '%s' header for Put Block List operation", name)
    with rest_utils.cloud_storage_http_request('put', request_url, data=data, headers=request_headers) as response:
        rest_utils.augmented_raise_for_status(response)