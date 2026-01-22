import base64
import contextlib
import re
from io import BytesIO
from . import lazy_import
from breezy import (
from breezy.bzr import (
from breezy.bzr.bundle import (
from . import errors, hooks, registry
def to_email(self, mail_to, branch, sign=False):
    """Serialize as an email message.

        :param mail_to: The address to mail the message to
        :param branch: The source branch, to get the signing strategy and
            source email address
        :param sign: If True, gpg-sign the email
        :return: an email message
        """
    mail_from = branch.get_config_stack().get('email')
    if self.message is not None:
        subject = self.message
    else:
        revision = branch.repository.get_revision(self.revision_id)
        subject = revision.message
    if sign:
        body = self.to_signed(branch)
    else:
        body = b''.join(self.to_lines())
    message = email_message.EmailMessage(mail_from, mail_to, subject, body)
    return message