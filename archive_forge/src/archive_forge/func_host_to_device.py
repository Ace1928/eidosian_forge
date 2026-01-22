def host_to_device(dst, src, size, stream=0):
    dst.view('u1')[:size] = src.view('u1')[:size]