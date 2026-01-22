def test_inet_pton_ipv4_valid(self):
    data = inet_pton(socket.AF_INET, '127.0.0.1')
    assert isinstance(data, bytes)