from __future__ import annotations
from email import message_from_bytes
from email.mime.message import MIMEMessage
from vine import promise, transform
from kombu.asynchronous.aws.ext import AWSRequest, get_response
from kombu.asynchronous.http import Headers, Request, get_client
def getrequest(self):
    headers = Headers(self.headers)
    return self.Request(self.path, method=self.method, headers=headers, body=self.body, connect_timeout=self.timeout, request_timeout=self.timeout, validate_cert=False)