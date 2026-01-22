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
def test_resolve_rule_list_populated(self):
    client_plugin, schema = self._test_resolve_rule(is_list=True)
    data = {'far': [{'red': 'blue'}, {'red': 'roses'}]}
    props = properties.Properties(schema, data)
    rule = translation.TranslationRule(props, translation.TranslationRule.RESOLVE, ['far', 'red'], client_plugin=client_plugin, finder='find_name_id')
    tran = translation.Translation(props)
    tran.set_rules([rule])
    self.assertTrue(tran.has_translation('far.red'))
    result = tran.translate('far.0.red', data['far'][0]['red'])
    self.assertEqual('yellow', result)
    self.assertEqual('yellow', tran.resolved_translations['far.0.red'])