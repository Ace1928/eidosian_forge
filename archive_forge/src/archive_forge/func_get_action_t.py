from ctypes import Structure, c_int, c_byte
def get_action_t(action, value):
    l_v = len(value)
    value = bytearray(value) + bytearray(b'\x00' * (_VALUE_BUFFER_SIZE - len(value)))
    value = (c_byte * _VALUE_BUFFER_SIZE).from_buffer(value)
    assert _VALUE_BUFFER_SIZE - len(value) >= 0
    return action_t(action, l_v, value)