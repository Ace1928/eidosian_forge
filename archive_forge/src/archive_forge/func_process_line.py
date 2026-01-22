import os
from ..controldir import ControlDir
from ..errors import NoRepositoryPresent, NotBranchError
from ..plugins.fastimport import exporter as fastexporter
from ..repository import InterRepository
from ..transport import get_transport_from_path
from . import LocalGitProber
from .dir import BareLocalGitControlDirFormat, LocalGitControlDirFormat
from .object_store import get_object_store
from .refs import get_refs_container, ref_to_branch_name
from .repository import GitRepository
def process_line(self, l, outf):
    argv = l.strip().split()
    if argv == []:
        if self.batchcmd == 'fetch':
            fetch(outf, self.wants, self.shortname, self.remote_dir, self.local_dir)
        elif self.batchcmd == 'push':
            push(outf, self.wants, self.shortname, self.remote_dir, self.local_dir)
        elif self.batchcmd is None:
            return
        else:
            raise AssertionError('invalid batch %r' % self.batchcmd)
        self.batchcmd = None
    else:
        try:
            self.commands[argv[0].decode()](self, outf, argv)
        except KeyError:
            raise Exception('Unknown remote command %r' % argv)
    outf.flush()