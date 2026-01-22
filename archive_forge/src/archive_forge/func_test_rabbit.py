from oslo_config import cfg
from oslo_config import types
from oslo_messaging._drivers import common as drv_cmn
from oslo_messaging.tests import utils as test_utils
from oslo_messaging import transport
def test_rabbit(self):
    group = 'oslo_messaging_rabbit'
    self.config(rabbit_retry_interval=1, rabbit_qos_prefetch_count=0, group=group)
    dummy_opts = [cfg.ListOpt('list_str', item_type=types.String(), default=[]), cfg.ListOpt('list_int', item_type=types.Integer(), default=[]), cfg.DictOpt('dict', default={}), cfg.BoolOpt('bool', default=False), cfg.StrOpt('str', default='default')]
    self.conf.register_opts(dummy_opts, group=group)
    url = transport.TransportURL.parse(self.conf, 'rabbit:///?rabbit_qos_prefetch_count=2&list_str=1&list_str=2&list_str=3&list_int=1&list_int=2&list_int=3&dict=x:1&dict=y:2&dict=z:3&bool=True')
    conf = drv_cmn.ConfigOptsProxy(self.conf, url, group)
    self.assertRaises(cfg.NoSuchOptError, conf.__getattr__, 'unknown_group')
    self.assertIsInstance(getattr(conf, group), conf.GroupAttrProxy)
    self.assertEqual(1, conf.oslo_messaging_rabbit.rabbit_retry_interval)
    self.assertEqual(2, conf.oslo_messaging_rabbit.rabbit_qos_prefetch_count)
    self.assertEqual(['1', '2', '3'], conf.oslo_messaging_rabbit.list_str)
    self.assertEqual([1, 2, 3], conf.oslo_messaging_rabbit.list_int)
    self.assertEqual({'x': '1', 'y': '2', 'z': '3'}, conf.oslo_messaging_rabbit.dict)
    self.assertEqual(True, conf.oslo_messaging_rabbit.bool)
    self.assertEqual('default', conf.oslo_messaging_rabbit.str)