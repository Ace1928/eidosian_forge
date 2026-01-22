import unittest
from nltk import tgrep
from nltk.tree import ParentedTree
def test_tokenize_segmented_patterns(self):
    """Test tokenization of segmented patterns."""
    self.assertEqual(tgrep.tgrep_tokenize('S < @SBJ=s < (@VP=v < (@VB $.. @OBJ)) : =s .. =v'), ['S', '<', '@SBJ', '=', 's', '<', '(', '@VP', '=', 'v', '<', '(', '@VB', '$..', '@OBJ', ')', ')', ':', '=s', '..', '=v'])