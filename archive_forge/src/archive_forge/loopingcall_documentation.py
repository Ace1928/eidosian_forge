import logging
import sys
from eventlet import event
from eventlet import greenthread
from oslo_utils import timeutils
:param retvalue: Value that LoopingCall.wait() should return.