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
def test_replace_rule_str_value_path_error(self):
    schema = {'far': properties.Schema(properties.Schema.STRING), 'bar': properties.Schema(properties.Schema.STRING)}
    data = {'far': 'one', 'bar': 'two'}
    props = properties.Properties(schema, data)
    rule = translation.TranslationRule(props, translation.TranslationRule.REPLACE, ['bar'], value_path=['far'])
    tran = translation.Translation(props)
    tran.set_rules([rule])
    self.assertTrue(tran.has_translation('bar'))
    ex = self.assertRaises(exception.StackValidationFailed, tran.translate, 'bar', data['bar'])
    self.assertEqual('Cannot define the following properties at the same time: bar, far', str(ex))