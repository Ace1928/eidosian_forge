import os
import stat
import sys
import time
import warnings
from io import BytesIO
from typing import (
from .errors import (
from .file import GitFile
from .hooks import (
from .line_ending import BlobNormalizer, TreeBlobNormalizer
from .object_store import (
from .objects import (
from .pack import generate_unpacked_objects
from .refs import (
def get_user_identity(config: 'StackedConfig', kind: Optional[str]=None) -> bytes:
    """Determine the identity to use for new commits.

    If kind is set, this first checks
    GIT_${KIND}_NAME and GIT_${KIND}_EMAIL.

    If those variables are not set, then it will fall back
    to reading the user.name and user.email settings from
    the specified configuration.

    If that also fails, then it will fall back to using
    the current users' identity as obtained from the host
    system (e.g. the gecos field, $EMAIL, $USER@$(hostname -f).

    Args:
      kind: Optional kind to return identity for,
        usually either "AUTHOR" or "COMMITTER".

    Returns:
      A user identity
    """
    user: Optional[bytes] = None
    email: Optional[bytes] = None
    if kind:
        user_uc = os.environ.get('GIT_' + kind + '_NAME')
        if user_uc is not None:
            user = user_uc.encode('utf-8')
        email_uc = os.environ.get('GIT_' + kind + '_EMAIL')
        if email_uc is not None:
            email = email_uc.encode('utf-8')
    if user is None:
        try:
            user = config.get(('user',), 'name')
        except KeyError:
            user = None
    if email is None:
        try:
            email = config.get(('user',), 'email')
        except KeyError:
            email = None
    default_user, default_email = _get_default_identity()
    if user is None:
        user = default_user.encode('utf-8')
    if email is None:
        email = default_email.encode('utf-8')
    if email.startswith(b'<') and email.endswith(b'>'):
        email = email[1:-1]
    return user + b' <' + email + b'>'