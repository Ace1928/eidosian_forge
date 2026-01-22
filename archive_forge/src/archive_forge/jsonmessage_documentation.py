import base64
from boto.sqs.message import MHMessage
from boto.exception import SQSDecodeError
from boto.compat import json

    Acts like a dictionary but encodes it's data as a Base64 encoded JSON payload.
    