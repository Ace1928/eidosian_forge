@given(port_numbers())
def test_port_numbers_bounds(self, port):
    """
            port_numbers() generates integers between 1 and 65535, inclusive.
            """
    self.assertGreaterEqual(port, 1)
    self.assertLessEqual(port, 65535)