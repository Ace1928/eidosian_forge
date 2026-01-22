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
def test_resolve_rule_other_with_get_attr(self):
    client_plugin, schema = self._test_resolve_rule()

    class DummyStack(dict):
        pass

    class rsrc(object):
        pass
    stack = DummyStack(another_res=rsrc())
    attr_func = cfn_funcs.GetAtt(stack, 'Fn::GetAtt', ['another_res', 'name'])
    data = {'far': attr_func}
    props = properties.Properties(schema, data)
    rule = translation.TranslationRule(props, translation.TranslationRule.RESOLVE, ['far'], client_plugin=client_plugin, finder='find_name_id')
    tran = translation.Translation(props)
    tran.set_rules([rule], client_resolve=False)
    self.assertFalse(tran.store_translated_values)
    self.assertFalse(tran.has_translation('far'))
    result = tran.translate('far', 'no_check', data['far'])
    self.assertEqual('no_check', result)
    self.assertIsNone(tran.resolved_translations.get('far'))