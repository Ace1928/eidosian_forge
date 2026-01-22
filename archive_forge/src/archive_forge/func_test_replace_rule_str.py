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
def test_replace_rule_str(self):
    schema = {'far': properties.Schema(properties.Schema.STRING), 'bar': properties.Schema(properties.Schema.STRING)}
    data = {'far': 'one', 'bar': 'two'}
    props = properties.Properties(schema, data)
    rule = translation.TranslationRule(props, translation.TranslationRule.REPLACE, ['bar'], props.get('far'))
    tran = translation.Translation(props)
    tran.set_rules([rule])
    self.assertTrue(tran.has_translation('bar'))
    result = tran.translate('bar', data['bar'])
    self.assertEqual('one', result)
    self.assertEqual('one', tran.resolved_translations['bar'])