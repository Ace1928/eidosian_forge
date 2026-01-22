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
def to_progress_instance(progress: Union[Callable[..., Any], RemoteProgress, None]) -> Union[RemoteProgress, CallableRemoteProgress]:
    """Given the 'progress' return a suitable object derived from RemoteProgress."""
    if callable(progress):
        return CallableRemoteProgress(progress)
    elif progress is None:
        return RemoteProgress()
    return progress