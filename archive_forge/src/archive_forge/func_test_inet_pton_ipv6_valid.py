def test_inet_pton_ipv6_valid(self):
    data = inet_pton(socket.AF_INET6, '::1')
    assert isinstance(data, bytes)