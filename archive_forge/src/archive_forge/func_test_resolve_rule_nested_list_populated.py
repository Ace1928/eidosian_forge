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
def test_resolve_rule_nested_list_populated(self):
    client_plugin, schema = self._test_resolve_rule_nested_list()
    data = {'instances': [{'networks': [{'port': 'port1', 'net': 'net1'}]}]}
    props = properties.Properties(schema, data)
    rule = translation.TranslationRule(props, translation.TranslationRule.RESOLVE, ['instances', 'networks', 'port'], client_plugin=client_plugin, finder='find_name_id', entity='port')
    tran = translation.Translation(props)
    tran.set_rules([rule])
    self.assertTrue(tran.has_translation('instances.networks.port'))
    result = tran.translate('instances.0.networks.0.port', data['instances'][0]['networks'][0]['port'])
    self.assertEqual('port1_id', result)
    self.assertEqual('port1_id', tran.resolved_translations['instances.0.networks.0.port'])