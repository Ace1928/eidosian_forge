from google.api_core import client_info
def to_grpc_metadata(self):
    """Returns the gRPC metadata for this client info."""
    return (METRICS_METADATA_KEY, self.to_user_agent())