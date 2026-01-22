import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def test_short_author(self):

    def assertAuthor(expected, author):
        self.rev.properties['author'] = author
        self.assertEqual(expected, self.lf.short_author(self.rev))
    assertAuthor('John Smith', 'John Smith <jsmith@example.com>')
    assertAuthor('John Smith', 'John Smith')
    assertAuthor('jsmith@example.com', 'jsmith@example.com')
    assertAuthor('jsmith@example.com', '<jsmith@example.com>')
    assertAuthor('John Smith', 'John Smith jsmith@example.com')