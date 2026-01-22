import os
import urllib
from typing import AnyStr
from twisted.internet import protocol
from twisted.logger import Logger
from twisted.python import filepath
from twisted.spread import pb
from twisted.web import http, resource, server, static
def view_resumeProducing(self, issuer):
    self.resumeProducing()