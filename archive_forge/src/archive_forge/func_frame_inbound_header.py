import zlib
from typing import Optional, Tuple, Union
from .frame_protocol import CloseReason, FrameDecoder, FrameProtocol, Opcode, RsvBits
def frame_inbound_header(self, proto: Union[FrameDecoder, FrameProtocol], opcode: Opcode, rsv: RsvBits, payload_length: int) -> Union[CloseReason, RsvBits]:
    if rsv.rsv1 and opcode.iscontrol():
        return CloseReason.PROTOCOL_ERROR
    if rsv.rsv1 and opcode is Opcode.CONTINUATION:
        return CloseReason.PROTOCOL_ERROR
    self._inbound_is_compressible = self._compressible_opcode(opcode)
    if self._inbound_compressed is None:
        self._inbound_compressed = rsv.rsv1
        if self._inbound_compressed:
            assert self._inbound_is_compressible
            if proto.client:
                bits = self.server_max_window_bits
            else:
                bits = self.client_max_window_bits
            if self._decompressor is None:
                self._decompressor = zlib.decompressobj(-int(bits))
    return RsvBits(True, False, False)