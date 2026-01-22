import re
import socket
import ssl
import boto
from boto.compat import six, http_client
Connect to a host on a given (SSL) port.