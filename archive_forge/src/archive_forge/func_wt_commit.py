import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def wt_commit(self, wt, message, **kwargs):
    """Use some mostly fixed values for commits to simplify tests.

        Tests can use this function to get some commit attributes. The time
        stamp is incremented at each commit.
        """
    if getattr(self, 'timestamp', None) is None:
        self.timestamp = 1132617600
    else:
        self.timestamp += 1
    kwargs.setdefault('timestamp', self.timestamp)
    kwargs.setdefault('timezone', 0)
    kwargs.setdefault('committer', 'Joe Foo <joe@foo.com>')
    return wt.commit(message, **kwargs)