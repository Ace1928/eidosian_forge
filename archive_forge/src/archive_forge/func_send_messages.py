import sys
import threading
from django.core.mail.backends.base import BaseEmailBackend
def send_messages(self, email_messages):
    """Write all messages to the stream in a thread-safe way."""
    if not email_messages:
        return
    msg_count = 0
    with self._lock:
        try:
            stream_created = self.open()
            for message in email_messages:
                self.write_message(message)
                self.stream.flush()
                msg_count += 1
            if stream_created:
                self.close()
        except Exception:
            if not self.fail_silently:
                raise
    return msg_count