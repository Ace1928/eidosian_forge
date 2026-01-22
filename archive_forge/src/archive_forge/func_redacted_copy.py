import threading
from oslo_config import cfg
from oslo_context.context import RequestContext
from oslo_utils import eventletutils
from oslotest import base
def redacted_copy(self):
    return self