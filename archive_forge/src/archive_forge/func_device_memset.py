def device_memset(dst, val, size, stream=0):
    dst.view('u1')[:size].fill(bytes([val])[0])