import zlib
from typing import Optional, Tuple, Union
from .frame_protocol import CloseReason, FrameDecoder, FrameProtocol, Opcode, RsvBits
def frame_inbound_complete(self, proto: Union[FrameDecoder, FrameProtocol], fin: bool) -> Union[bytes, CloseReason, None]:
    if not fin:
        return None
    if not self._inbound_is_compressible:
        self._inbound_compressed = None
        return None
    if not self._inbound_compressed:
        self._inbound_compressed = None
        return None
    assert self._decompressor is not None
    try:
        data = self._decompressor.decompress(b'\x00\x00\xff\xff')
        data += self._decompressor.flush()
    except zlib.error:
        return CloseReason.INVALID_FRAME_PAYLOAD_DATA
    if proto.client:
        no_context_takeover = self.server_no_context_takeover
    else:
        no_context_takeover = self.client_no_context_takeover
    if no_context_takeover:
        self._decompressor = None
    self._inbound_compressed = None
    return data