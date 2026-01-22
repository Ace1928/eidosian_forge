import logging
import urllib
from copy import deepcopy
from mlflow.utils import rest_utils
from mlflow.utils.file_utils import read_chunk
def patch_adls_flush(sas_url, position, headers):
    """Performs an ADLS Azure file flush `Patch` operation
    (https://docs.microsoft.com/en-us/rest/api/storageservices/datalakestoragegen2/path/update)

    Args:
        sas_url: A shared access signature URL referring to the Azure ADLS server
            to which the file update command should be issued.
        position: The final size of the file to flush.
        headers: Additional headers to include in the Patch request body.

    """
    request_url = _append_query_parameters(sas_url, {'action': 'flush', 'position': str(position)})
    request_headers = {}
    for name, value in headers.items():
        if _is_valid_adls_put_header(name):
            request_headers[name] = value
        else:
            _logger.debug("Removed unsupported '%s' header for ADLS Gen2 Patch operation", name)
    with rest_utils.cloud_storage_http_request('patch', request_url, headers=request_headers) as response:
        rest_utils.augmented_raise_for_status(response)