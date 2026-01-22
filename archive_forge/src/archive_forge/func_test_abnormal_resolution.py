from collections import defaultdict
import unittest
from lazr.uri import (
def test_abnormal_resolution(self):
    base = URI('http://a/b/c/d;p?q')

    def resolve(relative):
        return str(base.resolve(relative))
    self.assertEqual(resolve('../../../g'), 'http://a/g')
    self.assertEqual(resolve('../../../../g'), 'http://a/g')
    self.assertEqual(resolve('/./g'), 'http://a/g')
    self.assertEqual(resolve('/../g'), 'http://a/g')
    self.assertEqual(resolve('g.'), 'http://a/b/c/g.')
    self.assertEqual(resolve('.g'), 'http://a/b/c/.g')
    self.assertEqual(resolve('g..'), 'http://a/b/c/g..')
    self.assertEqual(resolve('..g'), 'http://a/b/c/..g')
    self.assertEqual(resolve('./../g'), 'http://a/b/g')
    self.assertEqual(resolve('./g/.'), 'http://a/b/c/g/')
    self.assertEqual(resolve('g/./h'), 'http://a/b/c/g/h')
    self.assertEqual(resolve('g/../h'), 'http://a/b/c/h')
    self.assertEqual(resolve('g;x=1/./y'), 'http://a/b/c/g;x=1/y')
    self.assertEqual(resolve('g;x=1/../y'), 'http://a/b/c/y')
    self.assertEqual(resolve('g?y/./x'), 'http://a/b/c/g?y/./x')
    self.assertEqual(resolve('g?y/../x'), 'http://a/b/c/g?y/../x')
    self.assertEqual(resolve('g#s/./x'), 'http://a/b/c/g#s/./x')
    self.assertEqual(resolve('g#s/../x'), 'http://a/b/c/g#s/../x')