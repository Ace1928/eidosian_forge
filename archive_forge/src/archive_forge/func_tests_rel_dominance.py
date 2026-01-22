import unittest
from nltk import tgrep
from nltk.tree import ParentedTree
def tests_rel_dominance(self):
    """
        Test matching nodes based on dominance relations.
        """
    tree = ParentedTree.fromstring('(S (A (T x)) (B (N x)))')
    self.assertEqual(list(tgrep.tgrep_positions('* < T', [tree])), [[(0,)]])
    self.assertEqual(list(tgrep.tgrep_positions('* < T > S', [tree])), [[(0,)]])
    self.assertEqual(list(tgrep.tgrep_positions('* !< T', [tree])), [[(), (0, 0), (0, 0, 0), (1,), (1, 0), (1, 0, 0)]])
    self.assertEqual(list(tgrep.tgrep_positions('* !< T > S', [tree])), [[(1,)]])
    self.assertEqual(list(tgrep.tgrep_positions('* > A', [tree])), [[(0, 0)]])
    self.assertEqual(list(tgrep.tgrep_positions('* > B', [tree])), [[(1, 0)]])
    self.assertEqual(list(tgrep.tgrep_positions('* !> B', [tree])), [[(), (0,), (0, 0), (0, 0, 0), (1,), (1, 0, 0)]])
    self.assertEqual(list(tgrep.tgrep_positions('* !> B >> S', [tree])), [[(0,), (0, 0), (1,)]])
    self.assertEqual(list(tgrep.tgrep_positions('* >> S', [tree])), [[(0,), (0, 0), (1,), (1, 0)]])
    self.assertEqual(list(tgrep.tgrep_positions('* >>, S', [tree])), [[(0,), (0, 0)]])
    self.assertEqual(list(tgrep.tgrep_positions("* >>' S", [tree])), [[(1,), (1, 0)]])
    self.assertEqual(list(tgrep.tgrep_positions('* << T', [tree])), [[(), (0,)]])
    self.assertEqual(list(tgrep.tgrep_positions("* <<' T", [tree])), [[(0,)]])
    self.assertEqual(list(tgrep.tgrep_positions('* <<1 N', [tree])), [[(1,)]])
    self.assertEqual(list(tgrep.tgrep_positions('* !<< T', [tree])), [[(0, 0), (0, 0, 0), (1,), (1, 0), (1, 0, 0)]])
    tree = ParentedTree.fromstring('(S (A (T x)) (B (T x) (N x )))')
    self.assertEqual(list(tgrep.tgrep_positions('* <: T', [tree])), [[(0,)]])
    self.assertEqual(list(tgrep.tgrep_positions('* < T', [tree])), [[(0,), (1,)]])
    self.assertEqual(list(tgrep.tgrep_positions('* !<: T', [tree])), [[(), (0, 0), (0, 0, 0), (1,), (1, 0), (1, 0, 0), (1, 1), (1, 1, 0)]])
    self.assertEqual(list(tgrep.tgrep_positions('* !<: T > S', [tree])), [[(1,)]])
    tree = ParentedTree.fromstring('(S (T (A x) (B x)) (T (C x)))')
    self.assertEqual(list(tgrep.tgrep_positions('* >: T', [tree])), [[(1, 0)]])
    self.assertEqual(list(tgrep.tgrep_positions('* !>: T', [tree])), [[(), (0,), (0, 0), (0, 0, 0), (0, 1), (0, 1, 0), (1,), (1, 0, 0)]])
    tree = ParentedTree.fromstring('(S (A (B (C (D (E (T x)))))) (A (B (C (D (E (T x))) (N x)))))')
    self.assertEqual(list(tgrep.tgrep_positions('* <<: T', [tree])), [[(0,), (0, 0), (0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0, 0)]])
    self.assertEqual(list(tgrep.tgrep_positions('* >>: A', [tree])), [[(0, 0), (0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0), (1, 0), (1, 0, 0)]])