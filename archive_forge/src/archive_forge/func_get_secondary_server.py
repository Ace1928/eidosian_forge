import base64
import re
from io import BytesIO
from urllib.request import parse_http_list, parse_keqv_list
from .. import errors, osutils, tests, transport
from ..bzr.smart import medium
from ..transport import chroot
from . import http_server
def get_secondary_server(self):
    """Get the server instance for the secondary transport."""
    if self.__secondary_server is None:
        self.__secondary_server = self.create_transport_secondary_server()
        self.start_server(self.__secondary_server)
    return self.__secondary_server