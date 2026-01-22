import ast
import tenacity
from oslo_log import log as logging
from heat.common import exception
from heat.objects import sync_point as sync_point_object
def str_pack_tuple(t):
    return u'tuple:' + str(tuple(t))