import copy
import queue
import threading
import time
from oslo_serialization import jsonutils
from oslo_utils import eventletutils
import oslo_messaging
from oslo_messaging._drivers import base
Make sure a message intended for rpc can be serialized.

        All the in tree drivers implementing RPC send uses jsonutils.dumps on
        the message. So in the test we ensure here that all the messages are
        serializable with this call.
        