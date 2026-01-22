import subprocess
import tempfile
from ... import errors
from ... import revision as _mod_revision
from ...config import ListOption, Option, bool_from_store, int_from_store
from ...email_message import EmailMessage
from ...smtp_connection import SMTPConnection
def subject(self):
    _subject = self.config.get('post_commit_subject')
    if _subject is None:
        _subject = 'Rev %d: %s in %s' % (self.revno, self.revision.get_summary(), self.url())
    return self._format(_subject)