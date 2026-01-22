from .waresponseparser import ResponseParser
from yowsup.env import YowsupEnv
import sys
import logging
from axolotl.ecc.curve import Curve
from axolotl.ecc.ec import ECPublicKey
from yowsup.common.tools import WATools
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from yowsup.config.v1.config import Config
from yowsup.profile.profile import YowProfile
import struct
import random
import base64
def sendPostRequest(self, parser=None):
    self.response = None
    params = self.params
    parser = parser or self.parser or ResponseParser()
    headers = dict(list({'User-Agent': self.getUserAgent(), 'Accept': parser.getMeta(), 'Content-Type': 'application/x-www-form-urlencoded'}.items()) + list(self.headers.items()))
    host, port, path = self.getConnectionParameters()
    self.response = WARequest.sendRequest(host, port, path, headers, params, 'POST')
    if not self.response.status == WARequest.OK:
        logger.error('Request not success, status was %s' % self.response.status)
        return {}
    data = self.response.read()
    logger.info(data)
    self.sent = True
    return parser.parse(data.decode(), self.pvars)