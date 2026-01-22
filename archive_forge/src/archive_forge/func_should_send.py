import subprocess
import tempfile
from ... import errors
from ... import revision as _mod_revision
from ...config import ListOption, Option, bool_from_store, int_from_store
from ...email_message import EmailMessage
from ...smtp_connection import SMTPConnection
def should_send(self):
    post_commit_push_pull = self.config.get('post_commit_push_pull')
    if post_commit_push_pull and self.op == 'commit':
        return False
    if not post_commit_push_pull and self.op != 'commit':
        return False
    return bool(self.to() and self.from_address())