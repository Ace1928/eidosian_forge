from collections import abc
import decimal
import random
import weakref
from oslo_log import log as logging
from oslo_utils import timeutils
from oslo_utils import uuidutils
from neutron_lib._i18n import _
def diff_list_of_dict(old_list, new_list):
    """Given 2 lists of dicts, return a tuple containing the diff.

    :param old_list: The old list of dicts to diff.
    :param new_list: The new list of dicts to diff.
    :returns: A tuple where the first item is a list of the added dicts in
        the diff and the second item is the removed dicts.
    """
    new_set = set([dict2str(i) for i in new_list])
    old_set = set([dict2str(i) for i in old_list])
    added = new_set - old_set
    removed = old_set - new_set
    return ([str2dict(a) for a in added], [str2dict(r) for r in removed])