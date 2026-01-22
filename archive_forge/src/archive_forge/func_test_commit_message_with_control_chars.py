import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def test_commit_message_with_control_chars(self):
    wt = self.make_branch_and_tree('.')
    msg = 'All 8-bit chars: ' + ''.join([chr(x) for x in range(256)])
    msg = msg.replace('\r', '\n')
    wt.commit(msg)
    lf = LogCatcher()
    log.show_log(wt.branch, lf, verbose=True)
    committed_msg = lf.revisions[0].rev.message
    if wt.branch.repository._serializer.squashes_xml_invalid_characters:
        self.assertNotEqual(msg, committed_msg)
        self.assertTrue(len(committed_msg) > len(msg))
    else:
        self.assertEqual(msg, committed_msg)