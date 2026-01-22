from ..lazy_import import lazy_import
import time
from breezy import (
from .. import lazy_regex
Return new revision-id.

    :param username: The username of the committer, in the format returned by
        config.username().  This is typically a real name, followed by an
        email address. If found, we will use just the email address portion.
        Otherwise we flatten the real name, and use that.
    :return: A new revision id.
    