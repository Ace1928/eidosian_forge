from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import textwrap
import unittest
from pasta.base import ast_utils
from pasta.base import scope
from pasta.base import test_utils
def test_top_level_imports(self):
    self.maxDiff = None
    source = textwrap.dedent('        import aaa\n        import bbb, ccc.ddd\n        import aaa.bbb.ccc\n        from eee import fff\n        from ggg.hhh import iii, jjj\n        ')
    tree = ast.parse(source)
    nodes = tree.body
    node_1_aaa = nodes[0].names[0]
    node_2_bbb = nodes[1].names[0]
    node_2_ccc_ddd = nodes[1].names[1]
    node_3_aaa_bbb_ccc = nodes[2].names[0]
    node_4_eee = nodes[3]
    node_4_fff = nodes[3].names[0]
    node_5_ggg_hhh = nodes[4]
    node_5_iii = nodes[4].names[0]
    node_5_jjj = nodes[4].names[1]
    s = scope.analyze(tree)
    self.assertItemsEqual(s.names.keys(), {'aaa', 'bbb', 'ccc', 'fff', 'iii', 'jjj'})
    self.assertItemsEqual(s.external_references.keys(), {'aaa', 'bbb', 'ccc', 'ccc.ddd', 'aaa.bbb', 'aaa.bbb.ccc', 'eee', 'eee.fff', 'ggg', 'ggg.hhh', 'ggg.hhh.iii', 'ggg.hhh.jjj'})
    self.assertItemsEqual(s.external_references['aaa'], [scope.ExternalReference('aaa', node_1_aaa, s.names['aaa']), scope.ExternalReference('aaa', node_3_aaa_bbb_ccc, s.names['aaa'])])
    self.assertItemsEqual(s.external_references['bbb'], [scope.ExternalReference('bbb', node_2_bbb, s.names['bbb'])])
    self.assertItemsEqual(s.external_references['ccc'], [scope.ExternalReference('ccc', node_2_ccc_ddd, s.names['ccc'])])
    self.assertItemsEqual(s.external_references['ccc.ddd'], [scope.ExternalReference('ccc.ddd', node_2_ccc_ddd, s.names['ccc'].attrs['ddd'])])
    self.assertItemsEqual(s.external_references['aaa.bbb'], [scope.ExternalReference('aaa.bbb', node_3_aaa_bbb_ccc, s.names['aaa'].attrs['bbb'])])
    self.assertItemsEqual(s.external_references['aaa.bbb.ccc'], [scope.ExternalReference('aaa.bbb.ccc', node_3_aaa_bbb_ccc, s.names['aaa'].attrs['bbb'].attrs['ccc'])])
    self.assertItemsEqual(s.external_references['eee'], [scope.ExternalReference('eee', node_4_eee, None)])
    self.assertItemsEqual(s.external_references['eee.fff'], [scope.ExternalReference('eee.fff', node_4_fff, s.names['fff'])])
    self.assertItemsEqual(s.external_references['ggg'], [scope.ExternalReference('ggg', node_5_ggg_hhh, None)])
    self.assertItemsEqual(s.external_references['ggg.hhh'], [scope.ExternalReference('ggg.hhh', node_5_ggg_hhh, None)])
    self.assertItemsEqual(s.external_references['ggg.hhh.iii'], [scope.ExternalReference('ggg.hhh.iii', node_5_iii, s.names['iii'])])
    self.assertItemsEqual(s.external_references['ggg.hhh.jjj'], [scope.ExternalReference('ggg.hhh.jjj', node_5_jjj, s.names['jjj'])])
    self.assertIs(s.names['aaa'].definition, node_1_aaa)
    self.assertIs(s.names['bbb'].definition, node_2_bbb)
    self.assertIs(s.names['ccc'].definition, node_2_ccc_ddd)
    self.assertIs(s.names['fff'].definition, node_4_fff)
    self.assertIs(s.names['iii'].definition, node_5_iii)
    self.assertIs(s.names['jjj'].definition, node_5_jjj)
    self.assertItemsEqual(s.names['aaa'].reads, [node_3_aaa_bbb_ccc])
    for ref in {'bbb', 'ccc', 'fff', 'iii', 'jjj'}:
        self.assertEqual(s.names[ref].reads, [], 'Expected no reads for %s' % ref)