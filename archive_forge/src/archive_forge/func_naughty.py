import oslo_messaging
from oslo_messaging.tests import utils as test_utils
@oslo_messaging.expected_exceptions(FooException)
def naughty():
    raise BarException()