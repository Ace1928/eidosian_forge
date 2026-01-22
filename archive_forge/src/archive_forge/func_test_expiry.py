import random
import time
import unittest
def test_expiry(self):

    @self._makeOne(1, None, timeout=0.1)
    def sleep_a_bit(param):
        time.sleep(0.1)
        return 2 * param
    start = time.time()
    result1 = sleep_a_bit('hello')
    stop = time.time()
    self.assertEqual(result1, 2 * 'hello')
    self.assertTrue(stop - start > 0.1)
    start = time.time()
    result2 = sleep_a_bit('hello')
    stop = time.time()
    self.assertEqual(result2, 2 * 'hello')
    self.assertTrue(stop - start < 0.1)
    time.sleep(0.1)
    start = time.time()
    result3 = sleep_a_bit('hello')
    stop = time.time()
    self.assertEqual(result3, 2 * 'hello')
    self.assertTrue(stop - start > 0.1)