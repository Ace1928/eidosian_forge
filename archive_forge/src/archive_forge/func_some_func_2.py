from unittest import mock
import ddt
import manilaclient
from manilaclient import api_versions
from manilaclient.common import cliutils
from manilaclient import exceptions
from manilaclient.tests.unit import utils
@cliutils.arg('name_2', help='Name of the something')
@cliutils.arg('action_2', help='Some action')
@api_versions.wraps('2.2', '2.6')
def some_func_2(cs, args):
    pass