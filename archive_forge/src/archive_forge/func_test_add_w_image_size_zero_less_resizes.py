import hashlib
import io
from unittest import mock
from oslo_utils.secretutils import md5
from oslo_utils import units
from glance_store._drivers import rbd as rbd_store
from glance_store import exceptions
from glance_store import location as g_location
from glance_store.tests import base
from glance_store.tests.unit import test_store_capabilities
from glance_store.tests import utils as test_utils
def test_add_w_image_size_zero_less_resizes(self):
    """Assert that correct size is returned even though 0 was provided."""
    data_len = 57 * units.Mi
    data_iter = test_utils.FakeData(data_len)
    with mock.patch.object(rbd_store.rbd.Image, 'resize') as resize:
        with mock.patch.object(rbd_store.rbd.Image, 'write') as write:
            ret = self.store.add('fake_image_id', data_iter, 0, self.hash_algo)
            expected = 1
            expected_calls = []
            data_len_temp = data_len
            resize_amount = self.store.WRITE_CHUNKSIZE
            while data_len_temp > 0:
                resize_amount *= 2
                expected_calls.append(resize_amount + (data_len - data_len_temp))
                data_len_temp -= resize_amount
                expected += 1
            self.assertEqual(expected, resize.call_count)
            resize.assert_has_calls([mock.call(call) for call in expected_calls])
            expected = [self.store.WRITE_CHUNKSIZE for i in range(int(data_len / self.store.WRITE_CHUNKSIZE))] + [data_len % self.store.WRITE_CHUNKSIZE]
            actual = [len(args[0]) for args, kwargs in write.call_args_list]
            self.assertEqual(expected, actual)
            self.assertEqual(data_len, resize.call_args_list[-1][0][0])
            self.assertEqual(data_len, ret[1])