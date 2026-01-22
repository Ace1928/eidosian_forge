from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import platform
import re
import six
import gslib
from gslib.cs_api_map import ApiSelector
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.utils.unit_util import ONE_KIB
def test_minus_D_cat(self):
    """Tests cat command with debug option."""
    key_uri = self.CreateObject(contents=b'0123456789')
    with SetBotoConfigForTest([('Boto', 'proxy_pass', 'secret')]):
        stdout, stderr = self.RunGsUtil(['-D', 'cat', suri(key_uri)], return_stdout=True, return_stderr=True)
    self.assertIn('You are running gsutil with debug output enabled.', stderr)
    self.assertIn('config:', stderr)
    if six.PY2:
        self.assertIn("('proxy_pass', u'REDACTED')", stderr)
    else:
        self.assertIn("('proxy_pass', 'REDACTED')", stderr)
    self.assertIn("reply: 'HTTP/1.1 200 OK", stderr)
    self.assert_header_in_output('Expires', None, stderr)
    self.assert_header_in_output('Date', None, stderr)
    self.assert_header_in_output('Content-Type', 'application/octet-stream', stderr)
    self.assert_header_in_output('Content-Length', '10', stderr)
    if self.test_api == ApiSelector.XML:
        self.assert_header_in_output('Cache-Control', None, stderr)
        self.assert_header_in_output('ETag', '"781e5e245d69b566979b86e28d23f2c7"', stderr)
        self.assert_header_in_output('Last-Modified', None, stderr)
        self.assert_header_in_output('x-goog-generation', None, stderr)
        self.assert_header_in_output('x-goog-metageneration', '1', stderr)
        self.assert_header_in_output('x-goog-hash', 'crc32c=KAwGng==', stderr)
        self.assert_header_in_output('x-goog-hash', 'md5=eB5eJF1ptWaXm4bijSPyxw==', stderr)
        regex_str = "send:\\s+([b|u]')?HEAD /%s/%s HTTP/[^\\\\]*\\\\r\\\\n(.*)" % (key_uri.bucket_name, key_uri.object_name)
        regex = re.compile(regex_str)
        match = regex.search(stderr)
        if not match:
            self.fail('Did not find this regex in stderr:\nRegex: %s\nStderr: %s' % (regex_str, stderr))
        request_fields_str = match.group(2)
        self.assertIn('Content-Length: 0', request_fields_str)
        self.assertRegex(request_fields_str, 'User-Agent: .*gsutil/%s.*interactive/False command/cat' % gslib.VERSION)
    elif self.test_api == ApiSelector.JSON:
        if six.PY2:
            self.assertIn("md5Hash: u'eB5eJF1ptWaXm4bijSPyxw=='", stderr)
        else:
            self.assertIn("md5Hash: 'eB5eJF1ptWaXm4bijSPyxw=='", stderr)
        self.assert_header_in_output('Cache-Control', 'no-cache, no-store, max-age=0, must-revalidate', stderr)
        self.assertRegex(stderr, '.*GET.*b/%s/o/%s' % (key_uri.bucket_name, key_uri.object_name))
        self.assertRegex(stderr, 'Python/%s.gsutil/%s.*interactive/False command/cat' % (platform.python_version(), gslib.VERSION))
    if gslib.IS_PACKAGE_INSTALL:
        self.assertIn('PACKAGED_GSUTIL_INSTALLS_DO_NOT_HAVE_CHECKSUMS', stdout)
    else:
        self.assertRegex(stdout, '.*checksum: [0-9a-f]{32}.*')
    self.assertIn('gsutil version: %s' % gslib.VERSION, stdout)
    self.assertIn('boto version: ', stdout)
    self.assertIn('python version: ', stdout)
    self.assertIn('OS: ', stdout)
    self.assertIn('multiprocessing available: ', stdout)
    self.assertIn('using cloud sdk: ', stdout)
    self.assertIn('pass cloud sdk credentials to gsutil: ', stdout)
    self.assertIn('config path(s): ', stdout)
    self.assertIn('gsutil path: ', stdout)
    self.assertIn('compiled crcmod: ', stdout)
    self.assertIn('installed via package manager: ', stdout)
    self.assertIn('editable install: ', stdout)