import errno
import smtplib
import socket
from email.utils import getaddresses, parseaddr
from . import config, osutils
from .errors import BzrError, InternalBzrError
Send an email message.

        The message will be sent to all addresses in the To, Cc and Bcc
        headers.

        :param message: An email.message.Message or
            email.mime.multipart.MIMEMultipart object.
        :return: None
        