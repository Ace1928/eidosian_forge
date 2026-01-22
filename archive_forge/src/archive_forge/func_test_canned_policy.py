import tempfile
import unittest
from boto.compat import StringIO, six, json
from textwrap import dedent
from boto.cloudfront.distribution import Distribution
def test_canned_policy(self):
    """
        Generate signed url from the Example Canned Policy in Amazon's
        documentation.
        """
    url = 'http://d604721fxaaqy9.cloudfront.net/horizon.jpg?large=yes&license=yes'
    expire_time = 1258237200
    expected_url = 'http://d604721fxaaqy9.cloudfront.net/horizon.jpg?large=yes&license=yes&Expires=1258237200&Signature=Nql641NHEUkUaXQHZINK1FZ~SYeUSoBJMxjdgqrzIdzV2gyEXPDNv0pYdWJkflDKJ3xIu7lbwRpSkG98NBlgPi4ZJpRRnVX4kXAJK6tdNx6FucDB7OVqzcxkxHsGFd8VCG1BkC-Afh9~lOCMIYHIaiOB6~5jt9w2EOwi6sIIqrg_&Key-Pair-Id=PK123456789754'
    signed_url = self.dist.create_signed_url(url, self.pk_id, expire_time, private_key_string=self.pk_str)
    self.assertEqual(expected_url, signed_url)