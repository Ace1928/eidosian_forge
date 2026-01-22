import uuid
import oauthlib.common
from oauthlib import oauth1
from oslo_log import log
from keystone.common import manager
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone import notifications
def set_verifier(self, verifier):
    self.verifier = verifier