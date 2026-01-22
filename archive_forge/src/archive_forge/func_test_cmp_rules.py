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
def test_cmp_rules(self):
    rules = [translation.TranslationRule(mock.Mock(spec=properties.Properties), translation.TranslationRule.DELETE, ['any']), translation.TranslationRule(mock.Mock(spec=properties.Properties), translation.TranslationRule.ADD, ['any']), translation.TranslationRule(mock.Mock(spec=properties.Properties), translation.TranslationRule.RESOLVE, ['any'], client_plugin=mock.ANY, finder=mock.ANY), translation.TranslationRule(mock.Mock(spec=properties.Properties), translation.TranslationRule.REPLACE, ['any'])]
    expected = [translation.TranslationRule.ADD, translation.TranslationRule.REPLACE, translation.TranslationRule.RESOLVE, translation.TranslationRule.DELETE]
    result = [rule.rule for rule in sorted(rules)]
    self.assertEqual(expected, result)