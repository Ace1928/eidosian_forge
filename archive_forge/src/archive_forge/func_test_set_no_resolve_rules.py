import copy
from unittest import mock
from heat.common import exception
from heat.engine.cfn import functions as cfn_funcs
from heat.engine import function
from heat.engine.hot import functions as hot_funcs
from heat.engine import parameters
from heat.engine import properties
from heat.engine import translation
from heat.tests import common
def test_set_no_resolve_rules(self):
    rules = [translation.TranslationRule(self.props, translation.TranslationRule.RESOLVE, ['a'], client_plugin=mock.ANY, finder='finder')]
    tran = translation.Translation()
    tran.set_rules(rules, client_resolve=False)
    self.assertEqual({}, tran._rules)