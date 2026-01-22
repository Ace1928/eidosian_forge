import tempfile
import unittest
from boto.compat import StringIO, six, json
from textwrap import dedent
from boto.cloudfront.distribution import Distribution
def test_sign_custom_policy_2(self):
    """
        Test signing custom policy 2 from amazon's cloudfront documentation.
        """
    expected = 'rc~5Qbbm8EJXjUTQ6Cn0LAxR72g1DOPrTmdtfbWVVgQNw0q~KHUAmBa2Zv1Wjj8dDET4XSL~Myh44CLQdu4dOH~N9huH7QfPSR~O4tIOS1WWcP~2JmtVPoQyLlEc8YHRCuN3nVNZJ0m4EZcXXNAS-0x6Zco2SYx~hywTRxWR~5Q_'
    sig = self.dist._sign_string(self.custom_policy_2, private_key_string=self.pk_str)
    encoded_sig = self.dist._url_base64_encode(sig)
    self.assertEqual(expected, encoded_sig)