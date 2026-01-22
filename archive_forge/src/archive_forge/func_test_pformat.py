import networkx as nx
import pickle
from taskflow import test
from taskflow.types import graph
from taskflow.types import sets
from taskflow.types import timing
from taskflow.types import tree
def test_pformat(self):
    root = tree.Node('CEO')
    expected = '\nCEO\n'
    self.assertEqual(expected.strip(), root.pformat())
    root.add(tree.Node('Infra'))
    expected = '\nCEO\n|__Infra\n'
    self.assertEqual(expected.strip(), root.pformat())
    root[0].add(tree.Node('Infra.1'))
    expected = '\nCEO\n|__Infra\n   |__Infra.1\n'
    self.assertEqual(expected.strip(), root.pformat())
    root.add(tree.Node('Mail'))
    expected = '\nCEO\n|__Infra\n|  |__Infra.1\n|__Mail\n'
    self.assertEqual(expected.strip(), root.pformat())
    root.add(tree.Node('Search'))
    expected = '\nCEO\n|__Infra\n|  |__Infra.1\n|__Mail\n|__Search\n'
    self.assertEqual(expected.strip(), root.pformat())
    root[-1].add(tree.Node('Search.1'))
    expected = '\nCEO\n|__Infra\n|  |__Infra.1\n|__Mail\n|__Search\n   |__Search.1\n'
    self.assertEqual(expected.strip(), root.pformat())
    root[-1].add(tree.Node('Search.2'))
    expected = '\nCEO\n|__Infra\n|  |__Infra.1\n|__Mail\n|__Search\n   |__Search.1\n   |__Search.2\n'
    self.assertEqual(expected.strip(), root.pformat())
    root[0].add(tree.Node('Infra.2'))
    expected = '\nCEO\n|__Infra\n|  |__Infra.1\n|  |__Infra.2\n|__Mail\n|__Search\n   |__Search.1\n   |__Search.2\n'
    self.assertEqual(expected.strip(), root.pformat())
    root[0].add(tree.Node('Infra.3'))
    expected = '\nCEO\n|__Infra\n|  |__Infra.1\n|  |__Infra.2\n|  |__Infra.3\n|__Mail\n|__Search\n   |__Search.1\n   |__Search.2\n'
    self.assertEqual(expected.strip(), root.pformat())
    root[0][-1].add(tree.Node('Infra.3.1'))
    expected = '\nCEO\n|__Infra\n|  |__Infra.1\n|  |__Infra.2\n|  |__Infra.3\n|     |__Infra.3.1\n|__Mail\n|__Search\n   |__Search.1\n   |__Search.2\n'
    self.assertEqual(expected.strip(), root.pformat())
    root[-1][0].add(tree.Node('Search.1.1'))
    expected = '\nCEO\n|__Infra\n|  |__Infra.1\n|  |__Infra.2\n|  |__Infra.3\n|     |__Infra.3.1\n|__Mail\n|__Search\n   |__Search.1\n   |  |__Search.1.1\n   |__Search.2\n'
    self.assertEqual(expected.strip(), root.pformat())
    root[1].add(tree.Node('Mail.1'))
    expected = '\nCEO\n|__Infra\n|  |__Infra.1\n|  |__Infra.2\n|  |__Infra.3\n|     |__Infra.3.1\n|__Mail\n|  |__Mail.1\n|__Search\n   |__Search.1\n   |  |__Search.1.1\n   |__Search.2\n'
    self.assertEqual(expected.strip(), root.pformat())
    root[1][0].add(tree.Node('Mail.1.1'))
    expected = '\nCEO\n|__Infra\n|  |__Infra.1\n|  |__Infra.2\n|  |__Infra.3\n|     |__Infra.3.1\n|__Mail\n|  |__Mail.1\n|     |__Mail.1.1\n|__Search\n   |__Search.1\n   |  |__Search.1.1\n   |__Search.2\n'
    self.assertEqual(expected.strip(), root.pformat())