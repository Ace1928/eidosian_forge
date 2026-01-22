@given(hostnames(allow_leading_digit=False, allow_idn=False))
def test_hostnames_ascii_nolead(self, hostname):
    """
            hostnames(allow_leading_digit=False, allow_idn=False) generates
            ASCII host names without leading digits.
            """
    self.assertTrue(hostname == hostname.lstrip(digits))