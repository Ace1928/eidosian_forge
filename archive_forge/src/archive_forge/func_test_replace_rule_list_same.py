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
def test_replace_rule_list_same(self):
    schema = {'far': properties.Schema(properties.Schema.LIST, schema=properties.Schema(properties.Schema.MAP, schema={'red': properties.Schema(properties.Schema.STRING), 'blue': properties.Schema(properties.Schema.STRING)}))}
    data = {'far': [{'blue': 'white'}, {'red': 'roses'}]}
    props = properties.Properties(schema, data)
    rule = translation.TranslationRule(props, translation.TranslationRule.REPLACE, ['far', 'red'], None, 'blue')
    tran = translation.Translation(props)
    tran.set_rules([rule])
    self.assertTrue(tran.has_translation('far.0.red'))
    result = tran.translate('far.0.red', data['far'][0].get('red'), data['far'][0])
    self.assertEqual('white', result)
    self.assertEqual('white', tran.resolved_translations['far.0.red'])
    self.assertIsNone(tran.resolved_translations['far.0.blue'])
    self.assertTrue(tran.has_translation('far.1.red'))
    result = tran.translate('far.1.red', data['far'][1]['red'], data['far'][1])
    self.assertEqual('roses', result)
    self.assertEqual('roses', tran.resolved_translations['far.1.red'])
    self.assertIsNone(tran.resolved_translations['far.1.blue'])