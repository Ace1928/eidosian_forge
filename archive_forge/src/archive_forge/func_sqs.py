from SQS when you have short-running tasks (or a large number of workers).
from __future__ import annotations
import base64
import socket
import string
import uuid
from datetime import datetime
from queue import Empty
from botocore.client import Config
from botocore.exceptions import ClientError
from vine import ensure_promise, promise, transform
from kombu.asynchronous import get_event_loop
from kombu.asynchronous.aws.ext import boto3, exceptions
from kombu.asynchronous.aws.sqs.connection import AsyncSQSConnection
from kombu.asynchronous.aws.sqs.message import AsyncMessage
from kombu.log import get_logger
from kombu.utils import scheduling
from kombu.utils.encoding import bytes_to_str, safe_str
from kombu.utils.json import dumps, loads
from kombu.utils.objects import cached_property
from . import virtual
def sqs(self, queue=None):
    if queue is not None and self.predefined_queues:
        if queue not in self.predefined_queues:
            raise UndefinedQueueException(f"Queue with name '{queue}' must be defined in 'predefined_queues'.")
        q = self.predefined_queues[queue]
        if self.transport_options.get('sts_role_arn'):
            return self._handle_sts_session(queue, q)
        if not self.transport_options.get('sts_role_arn'):
            if queue in self._predefined_queue_clients:
                return self._predefined_queue_clients[queue]
            else:
                c = self._predefined_queue_clients[queue] = self.new_sqs_client(region=q.get('region', self.region), access_key_id=q.get('access_key_id', self.conninfo.userid), secret_access_key=q.get('secret_access_key', self.conninfo.password))
                return c
    if self._sqs is not None:
        return self._sqs
    c = self._sqs = self.new_sqs_client(region=self.region, access_key_id=self.conninfo.userid, secret_access_key=self.conninfo.password)
    return c