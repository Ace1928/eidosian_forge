import pycurl, random, socket
def opensocket(curl, purpose, curl_address):
    if random.random() < 0.5:
        curl.exception = ConnectionRejected('Rejecting connection attempt in opensocket callback')
        return pycurl.SOCKET_BAD
    family, socktype, protocol, address = curl_address
    s = socket.socket(family, socktype, protocol)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    return s