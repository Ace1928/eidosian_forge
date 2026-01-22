from grpc import StatusCode
from google.api_core.exceptions import GoogleAPICallError
def is_retryable(error: GoogleAPICallError) -> bool:
    return error.grpc_status_code in retryable_codes