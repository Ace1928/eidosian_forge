from .exceptions import HPACKDecodingError
def decode_huffman(huffman_string):
    """
    Given a bytestring of Huffman-encoded data for HPACK, returns a bytestring
    of the decompressed data.
    """
    if not huffman_string:
        return b''
    state = 0
    flags = 0
    decoded_bytes = bytearray()
    huffman_string = bytearray(huffman_string)
    for input_byte in huffman_string:
        index = state * 16 + (input_byte >> 4)
        state, flags, output_byte = HUFFMAN_TABLE[index]
        if flags & HUFFMAN_FAIL:
            raise HPACKDecodingError('Invalid Huffman String')
        if flags & HUFFMAN_EMIT_SYMBOL:
            decoded_bytes.append(output_byte)
        index = state * 16 + (input_byte & 15)
        state, flags, output_byte = HUFFMAN_TABLE[index]
        if flags & HUFFMAN_FAIL:
            raise HPACKDecodingError('Invalid Huffman String')
        if flags & HUFFMAN_EMIT_SYMBOL:
            decoded_bytes.append(output_byte)
    if not flags & HUFFMAN_COMPLETE:
        raise HPACKDecodingError('Incomplete Huffman string')
    return bytes(decoded_bytes)