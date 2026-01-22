from .... import config, msgeditor
from ....tests import TestCaseWithTransport
from ... import commitfromnews
def test_initial(self):
    self.setup_capture()
    self.enable_commitfromnews()
    builder = self.make_branch_builder('test')
    builder.start_series()
    builder.build_snapshot(None, [('add', ('', None, 'directory', None)), ('add', ('foo', b'foo-id', 'file', b'a\nb\nc\nd\ne\n'))], message_callback=msgeditor.generate_commit_message_template, revision_id=b'BASE-id')
    builder.finish_series()
    self.assertEqual([None], self.messages)