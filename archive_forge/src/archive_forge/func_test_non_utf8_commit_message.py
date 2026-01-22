import time
from .... import tests
from ..helpers import kind_to_mode
from . import FastimportFeature
def test_non_utf8_commit_message(self):
    handler, branch = self.get_handler()

    def files_one():
        yield commands.FileModifyCommand(b'a', kind_to_mode('file', False), None, b'data')

    def command_list():
        committer = [b'', b'elmer@a.com', time.time(), time.timezone]
        yield commands.CommitCommand(b'head', b'1', None, committer, b'This is a funky character: \x83', None, [], files_one)
    handler.process(command_list)
    rev = branch.repository.get_revision(branch.last_revision())
    self.assertEqual('This is a funky character: �', rev.message)