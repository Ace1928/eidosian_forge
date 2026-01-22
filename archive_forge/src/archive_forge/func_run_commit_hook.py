from io import BytesIO
import os
import os.path as osp
from pathlib import Path
from stat import (
import subprocess
from git.cmd import handle_process_output, safer_popen
from git.compat import defenc, force_bytes, force_text, safe_decode
from git.exc import HookExecutionError, UnmergedEntriesError
from git.objects.fun import (
from git.util import IndexFileSHA1Writer, finalize_process
from gitdb.base import IStream
from gitdb.typ import str_tree_type
from .typ import BaseIndexEntry, IndexEntry, CE_NAMEMASK, CE_STAGESHIFT
from .util import pack, unpack
from typing import Dict, IO, List, Sequence, TYPE_CHECKING, Tuple, Type, Union, cast
from git.types import PathLike
def run_commit_hook(name: str, index: 'IndexFile', *args: str) -> None:
    """Run the commit hook of the given name. Silently ignore hooks that do not exist.

    :param name: name of hook, like 'pre-commit'
    :param index: IndexFile instance
    :param args: Arguments passed to hook file
    :raises HookExecutionError:
    """
    hp = hook_path(name, index.repo.git_dir)
    if not os.access(hp, os.X_OK):
        return
    env = os.environ.copy()
    env['GIT_INDEX_FILE'] = safe_decode(str(index.path))
    env['GIT_EDITOR'] = ':'
    cmd = [hp]
    try:
        if os.name == 'nt' and (not _has_file_extension(hp)):
            relative_hp = Path(hp).relative_to(index.repo.working_dir).as_posix()
            cmd = ['bash.exe', relative_hp]
        process = safer_popen(cmd + list(args), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=index.repo.working_dir)
    except Exception as ex:
        raise HookExecutionError(hp, ex) from ex
    else:
        stdout_list: List[str] = []
        stderr_list: List[str] = []
        handle_process_output(process, stdout_list.append, stderr_list.append, finalize_process)
        stdout = ''.join(stdout_list)
        stderr = ''.join(stderr_list)
        if process.returncode != 0:
            stdout = force_text(stdout, defenc)
            stderr = force_text(stderr, defenc)
            raise HookExecutionError(hp, process.returncode, stderr, stdout)