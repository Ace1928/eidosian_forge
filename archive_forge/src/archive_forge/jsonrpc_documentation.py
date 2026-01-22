import codecs
import errno
import os
import random
import sys
import ovs.json
import ovs.poller
import ovs.reconnect
import ovs.stream
import ovs.timeval
import ovs.util
import ovs.vlog
 Resets the reconnect backoff by allowing as many free tries as the
        number of configured remotes.  This is to be used by upper layers
        before calling force_reconnect() if backoff is undesirable.