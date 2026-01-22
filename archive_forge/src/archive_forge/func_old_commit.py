import contextlib
import logging
import re
from git.cmd import Git, handle_process_output
from git.compat import defenc, force_text
from git.config import GitConfigParser, SectionConstraint, cp
from git.exc import GitCommandError
from git.refs import Head, Reference, RemoteReference, SymbolicReference, TagReference
from git.util import (
from typing import (
from git.types import PathLike, Literal, Commit_ish
@property
def old_commit(self) -> Union[str, SymbolicReference, Commit_ish, None]:
    return self._old_commit_sha and self._remote.repo.commit(self._old_commit_sha) or None