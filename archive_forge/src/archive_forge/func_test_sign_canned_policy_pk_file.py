import tempfile
import unittest
from boto.compat import StringIO, six, json
from textwrap import dedent
from boto.cloudfront.distribution import Distribution
def test_sign_canned_policy_pk_file(self):
    """
        Test signing the canned policy from amazon's cloudfront documentation
        with a file object.
        """
    expected = 'Nql641NHEUkUaXQHZINK1FZ~SYeUSoBJMxjdgqrzIdzV2gyEXPDNv0pYdWJkflDKJ3xIu7lbwRpSkG98NBlgPi4ZJpRRnVX4kXAJK6tdNx6FucDB7OVqzcxkxHsGFd8VCG1BkC-Afh9~lOCMIYHIaiOB6~5jt9w2EOwi6sIIqrg_'
    pk_file = tempfile.TemporaryFile()
    pk_file.write(self.pk_str)
    pk_file.seek(0)
    sig = self.dist._sign_string(self.canned_policy, private_key_file=pk_file)
    encoded_sig = self.dist._url_base64_encode(sig)
    self.assertEqual(expected, encoded_sig)