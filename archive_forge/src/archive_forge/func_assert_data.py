from tests.unit import unittest
import boto.swf.layer1_decisions
def assert_data(self, *data):
    self.assertEquals(self.decisions._data, list(data))