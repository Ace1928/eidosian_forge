from google.protobuf import message
import requests
from absl import logging
from tensorboard import version
from tensorboard.plugins.scalar import metadata as scalars_metadata
from tensorboard.uploader.proto import server_info_pb2
def create_server_info(frontend_origin, api_endpoint, upload_plugins):
    """Manually creates server info given a frontend and backend.

    Args:
      frontend_origin: The origin of the TensorBoard.dev frontend, like
        "https://tensorboard.dev" or "http://localhost:8000".
      api_endpoint: As to `server_info_pb2.ApiServer.endpoint`.
      upload_plugins: List of plugin names requested by the user and to be
        verified by the server.

    Returns:
      A `server_info_pb2.ServerInfoResponse` message.
    """
    result = server_info_pb2.ServerInfoResponse()
    result.compatibility.verdict = server_info_pb2.VERDICT_OK
    result.api_server.endpoint = api_endpoint
    url_format = result.url_format
    placeholder = '{{EID}}'
    while placeholder in frontend_origin:
        placeholder = '{%s}' % placeholder
    url_format.template = '%s/experiment/%s/' % (frontend_origin, placeholder)
    url_format.id_placeholder = placeholder
    result.plugin_control.allowed_plugins[:] = upload_plugins
    return result