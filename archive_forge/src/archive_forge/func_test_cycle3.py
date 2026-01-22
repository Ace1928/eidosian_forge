import unittest
from traits.api import (
def test_cycle3(self):
    lt = LinkTest(tc=self, head=self.build_list())
    handlers = [lt.arg_check0, lt.arg_check3, lt.arg_check4]
    nh = len(handlers)
    self.multi_register(lt, handlers, 'head.next*.value')
    link = self.new_link(lt, lt.head, 1)
    self.assertEqual(lt.calls, nh)
    link = self.new_link(lt, link, 2)
    self.assertEqual(lt.calls, 2 * nh)
    self.multi_register(lt, handlers, 'head.next*.value', remove=True)
    link = self.new_link(lt, link, 3)
    self.assertEqual(lt.calls, 2 * nh)