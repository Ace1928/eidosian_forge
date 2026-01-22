import copy
import itertools
from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient import config
from manilaclient import exceptions
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import share_types
def get_valid_type_create_data_2_0():
    public = [True, False]
    dhss = [True, False]
    snapshot = [None, True, False]
    extra_specs = [None, {'foo': 'bar'}]
    combos = list(itertools.product(public, dhss, snapshot, extra_specs))
    return combos