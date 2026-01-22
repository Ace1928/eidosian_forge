import zlib
from typing import Optional, Tuple, Union
from .frame_protocol import CloseReason, FrameDecoder, FrameProtocol, Opcode, RsvBits
def frame_outbound(self, proto: Union[FrameDecoder, FrameProtocol], opcode: Opcode, rsv: RsvBits, data: bytes, fin: bool) -> Tuple[RsvBits, bytes]:
    if not self._compressible_opcode(opcode):
        return (rsv, data)
    if opcode is not Opcode.CONTINUATION:
        rsv = RsvBits(True, *rsv[1:])
    if self._compressor is None:
        assert opcode is not Opcode.CONTINUATION
        if proto.client:
            bits = self.client_max_window_bits
        else:
            bits = self.server_max_window_bits
        self._compressor = zlib.compressobj(zlib.Z_DEFAULT_COMPRESSION, zlib.DEFLATED, -int(bits))
    data = self._compressor.compress(bytes(data))
    if fin:
        data += self._compressor.flush(zlib.Z_SYNC_FLUSH)
        data = data[:-4]
        if proto.client:
            no_context_takeover = self.client_no_context_takeover
        else:
            no_context_takeover = self.server_no_context_takeover
        if no_context_takeover:
            self._compressor = None
    return (rsv, data)