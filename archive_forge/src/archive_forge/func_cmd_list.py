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
def cmd_list(self, outf, argv):
    try:
        repo = self.remote_dir.find_repository()
    except NoRepositoryPresent:
        repo = self.remote_dir.create_repository()
    object_store = get_object_store(repo)
    with object_store.lock_read():
        refs = get_refs_container(self.remote_dir, object_store)
        for ref, git_sha1 in refs.as_dict().items():
            ref = ref.replace(b'~', b'_')
            outf.write(b'%s %s\n' % (git_sha1, ref))
        outf.write(b'\n')