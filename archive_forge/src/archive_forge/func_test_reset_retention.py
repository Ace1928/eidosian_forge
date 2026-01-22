from io import BytesIO
from unittest import TestCase
from fastimport import (
from fastimport.processors import (
from :2
from :2
from :100
from :101
from :100
from :100
from :100
from :100
from :101
from :100
from :100
from :102
from :102
from :102
from :100
from :102
from :100
from :102
from :100
from :102
from :102
from :102
from :100
from :102
from :100
from :100
from :100
from :100
from :100
from :102
from :101
from :102
from :101
import
from :999
from :3
import
from :999
from :3
import
from :999
from :3
import
from :999
from :3
import
from :999
from :3
def test_reset_retention(self):
    params = {b'include_paths': [b'NEWS']}
    self.assertFiltering(_SAMPLE_WITH_RESETS, params, b'blob\nmark :2\ndata 17\nLife\nis\ngood ...\ncommit refs/heads/master\nmark :101\ncommitter a <b@c> 1234798653 +0000\ndata 8\ntest\ning\nM 644 :2 NEWS\nreset refs/heads/foo\nreset refs/heads/bar\nfrom :101\n')