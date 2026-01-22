import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def test_properties_handler_bad_argument(self):
    wt = self.make_standard_commit('bad_argument', revprops={'a_prop': 'test_value'})
    sio = self.make_utf8_encoded_stringio()
    formatter = log.LongLogFormatter(to_file=sio)

    def bad_argument_prop_handler(revision):
        return {'custom_prop_name': revision.properties['a_prop']}
    log.properties_handler_registry.register('bad_argument_prop_handler', bad_argument_prop_handler)
    self.assertRaises(AttributeError, formatter.show_properties, 'a revision', '')
    revision = wt.branch.repository.get_revision(wt.branch.last_revision())
    formatter.show_properties(revision, '')
    self.assertEqualDiff(b'custom_prop_name: test_value\n', sio.getvalue())