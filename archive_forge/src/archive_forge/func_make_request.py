from io import BytesIO
from ...bzr.smart import medium
from ...transport import chroot, get_transport
from ...urlutils import local_path_to_url
def make_request(self, transport, write_func, request_bytes, rcp):
    protocol_factory, unused_bytes = medium._get_protocol_factory_for_bytes(request_bytes)
    server_protocol = protocol_factory(transport, write_func, rcp, self.backing_transport)
    server_protocol.accept_bytes(unused_bytes)
    return server_protocol