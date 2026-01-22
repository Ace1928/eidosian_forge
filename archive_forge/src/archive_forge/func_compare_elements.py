from collections import abc
import decimal
import random
import weakref
from oslo_log import log as logging
from oslo_utils import timeutils
from oslo_utils import uuidutils
from neutron_lib._i18n import _
def compare_elements(a, b):
    """Compare elements if a and b have same elements.

    This method doesn't consider ordering.

    :param a: The first item to compare.
    :param b: The second item to compare.
    :returns: True if a and b have the same elements, False otherwise.
    """
    return set(a or []) == set(b or [])