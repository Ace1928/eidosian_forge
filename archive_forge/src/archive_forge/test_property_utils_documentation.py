from glance.api import policy
from glance.common import exception
from glance.common import property_utils
import glance.context
from glance.tests.unit import base

        Verify rules are iterable in the same order as read from the config
        file
        