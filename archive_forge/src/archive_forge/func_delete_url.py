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
def delete_url(self, url: str, **kwargs: Any) -> 'Remote':
    """Deletes a new url on current remote (special case of git remote set_url)

        This command deletes new URLs to a given remote, making it possible to have
        multiple URLs for a single remote.

        :param url: String being the URL to delete from the remote
        :return: self
        """
    return self.set_url(url, delete=True)