from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import http_wrapper as apitools_http_wrapper
from googlecloudsdk.api_lib.storage import errors as cloud_errors
from googlecloudsdk.api_lib.storage import retry_util
from googlecloudsdk.calliope import exceptions as calliope_errors
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
import oauth2client
def launch_retriable(download_stream, apitools_download, start_byte=0, end_byte=None, additional_headers=None):
    """Wraps download to make it retriable."""
    progress_state = {'start_byte': start_byte}
    retry_util.set_retry_func(apitools_download)

    def _should_retry_resumable_download(exc_type, exc_value, exc_traceback, state):
        converted_error, _ = calliope_errors.ConvertKnownError(exc_value)
        if isinstance(exc_value, oauth2client.client.HttpAccessTokenRefreshError):
            if exc_value.status < 500 and exc_value.status != 429:
                return False
        elif not (isinstance(converted_error, core_exceptions.NetworkIssueError) or isinstance(converted_error, cloud_errors.RetryableApiError)):
            return False
        start_byte = download_stream.tell()
        if start_byte > progress_state['start_byte']:
            progress_state['start_byte'] = start_byte
            state.retrial = 0
        log.debug('Retrying download from byte {} after exception: {}. Trace: {}'.format(start_byte, exc_type, exc_traceback))
        apitools_http_wrapper.RebuildHttpConnections(apitools_download.bytes_http)
        return True

    def _call_launch():
        return launch(apitools_download, start_byte=progress_state['start_byte'], end_byte=end_byte, additional_headers=additional_headers)
    return retry_util.retryer(target=_call_launch, should_retry_if=_should_retry_resumable_download)