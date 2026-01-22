import unittest
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from apitools.base.py import exceptions
from apitools.base.py import util
def testMapRequestParams(self):
    params = {'str_field': 'foo', 'enum_field': MessageWithRemappings.AnEnum.value_one, 'enum_field_remapping': MessageWithRemappings.AnEnum.value_one}
    remapped_params = {'path_field': 'foo', 'enum_field': 'ONE', 'enum_field_remapped': 'ONE'}
    self.assertEqual(remapped_params, util.MapRequestParams(params, MessageWithRemappings))
    params['enum_field'] = MessageWithRemappings.AnEnum.value_two
    remapped_params['enum_field'] = 'value_two'
    self.assertEqual(remapped_params, util.MapRequestParams(params, MessageWithRemappings))