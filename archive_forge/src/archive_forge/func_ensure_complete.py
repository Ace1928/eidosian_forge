import random
import email.message
import pyzor
def ensure_complete(self):
    if 'Op' not in self:
        raise pyzor.IncompleteMessageError("doesn't have fields for a Request")
    ThreadedMessage.ensure_complete(self)