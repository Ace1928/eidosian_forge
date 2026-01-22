from .... import config, msgeditor
from ....tests import TestCaseWithTransport
from ... import commitfromnews
def capture_template(self, commit, message):
    self.commits.append(commit)
    self.messages.append(message)
    if message is None:
        message = 'let this commit succeed I command thee.'
    return message