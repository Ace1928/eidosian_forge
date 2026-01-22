from pprint import pformat
from six import iteritems
import re
@git_commit.setter
def git_commit(self, git_commit):
    """
        Sets the git_commit of this VersionInfo.

        :param git_commit: The git_commit of this VersionInfo.
        :type: str
        """
    if git_commit is None:
        raise ValueError('Invalid value for `git_commit`, must not be `None`')
    self._git_commit = git_commit