import os
import sys
from .. import bedding, osutils, tests
def test_default_mail_domain_simple(self):
    with open('simple', 'w') as f:
        f.write('domainname.com\n')
    r = bedding._get_default_mail_domain('simple')
    self.assertEqual('domainname.com', r)