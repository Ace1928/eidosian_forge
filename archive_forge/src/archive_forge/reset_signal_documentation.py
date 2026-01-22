from google.api_core.exceptions import GoogleAPICallError
from google.cloud.pubsublite.internal.status_codes import is_retryable
from google.rpc.error_details_pb2 import ErrorInfo
from google.rpc import status_pb2

    Determines whether the given error contains the stream RESET signal, sent by
    the server to instruct clients to reset stream state.

    Returns: True if the error contains the RESET signal.
    