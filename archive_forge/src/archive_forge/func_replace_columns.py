import argparse
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
@staticmethod
def replace_columns(cols, rules, reverse=False):
    if reverse:
        rules = dict(((rules[k], k) for k in rules.keys()))
    return [rules.get(col, col) for col in cols]