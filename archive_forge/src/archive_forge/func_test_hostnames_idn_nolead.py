@given(hostnames(allow_leading_digit=False))
def test_hostnames_idn_nolead(self, hostname):
    """
            hostnames(allow_leading_digit=False) generates a IDN host names
            without leading digits.
            """
    self.assertTrue(hostname == hostname.lstrip(digits))