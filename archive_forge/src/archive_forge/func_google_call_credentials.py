import threading  # pylint: disable=unused-import
import grpc
from grpc import _auth
from grpc.beta import _client_adaptations
from grpc.beta import _metadata
from grpc.beta import _server_adaptations
from grpc.beta import interfaces  # pylint: disable=unused-import
from grpc.framework.common import cardinality  # pylint: disable=unused-import
from grpc.framework.interfaces.face import face  # pylint: disable=unused-import
def google_call_credentials(credentials):
    """Construct CallCredentials from GoogleCredentials.

    Args:
      credentials: A GoogleCredentials object from the oauth2client library.

    Returns:
      A CallCredentials object for use in a GRPCCallOptions object.
    """
    return metadata_call_credentials(_auth.GoogleCallCredentials(credentials))