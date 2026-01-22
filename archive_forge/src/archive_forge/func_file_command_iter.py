import time
from .... import tests
from ..helpers import kind_to_mode
from . import FastimportFeature
def file_command_iter(self):

    def command_list():
        committer_a = [b'', b'a@elmer.com', time.time(), time.timezone]

        def files_one():
            yield commands.FileModifyCommand(b'foo\x83', kind_to_mode('file', False), None, b'content A\n')
        yield commands.CommitCommand(b'head', b'1', None, committer_a, b'commit 1', None, [], files_one)
    return command_list