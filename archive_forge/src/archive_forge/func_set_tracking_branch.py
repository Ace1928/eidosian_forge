from git.config import GitConfigParser, SectionConstraint
from git.util import join_path
from git.exc import GitCommandError
from .symbolic import SymbolicReference
from .reference import Reference
from typing import Any, Sequence, Union, TYPE_CHECKING
from git.types import PathLike, Commit_ish
def set_tracking_branch(self, remote_reference: Union['RemoteReference', None]) -> 'Head':
    """Configure this branch to track the given remote reference. This will
        alter this branch's configuration accordingly.

        :param remote_reference: The remote reference to track or None to untrack
            any references.
        :return: self
        """
    from .remote import RemoteReference
    if remote_reference is not None and (not isinstance(remote_reference, RemoteReference)):
        raise ValueError('Incorrect parameter type: %r' % remote_reference)
    with self.config_writer() as writer:
        if remote_reference is None:
            writer.remove_option(self.k_config_remote)
            writer.remove_option(self.k_config_remote_ref)
            if len(writer.options()) == 0:
                writer.remove_section()
        else:
            writer.set_value(self.k_config_remote, remote_reference.remote_name)
            writer.set_value(self.k_config_remote_ref, Head.to_full_path(remote_reference.remote_head))
    return self