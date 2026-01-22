import os
import signal
import time
import fixtures
from pifpaf.drivers import rabbitmq
from oslo_messaging.tests.functional import utils
from oslo_messaging.tests import utils as test_utils
def just_process(self, *args, **kargs):
    return 'callback done'