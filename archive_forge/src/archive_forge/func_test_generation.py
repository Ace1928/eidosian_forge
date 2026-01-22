from .protocoltreenode import ProtocolTreeNode
import unittest, time
def test_generation(self):
    if self.ProtocolEntity is None:
        raise ValueError('Test case not setup!')
    entity = self.ProtocolEntity.fromProtocolTreeNode(self.node)
    try:
        self.assertEqual(entity.toProtocolTreeNode(), self.node)
    except:
        print(entity.toProtocolTreeNode())
        print('\nNOTEQ\n')
        print(self.node)
        raise