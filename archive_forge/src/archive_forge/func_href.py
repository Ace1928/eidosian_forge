import re
from zaqarclient._i18n import _  # noqa
from zaqarclient import errors
from zaqarclient.queues.v1 import claim as claim_api
from zaqarclient.queues.v1 import core
from zaqarclient.queues.v1 import iterator
from zaqarclient.queues.v1 import message
@property
def href(self):
    return self._href