import unittest
from nltk.metrics.agreement import AnnotationTask
def test_easy2(self):
    """
        Same simple test with 1 rating removed.
        Removal of that rating should not matter: K-Apha ignores items with
        only 1 rating.
        """
    data = [('coder1', 'dress1', 'YES'), ('coder2', 'dress1', 'NO'), ('coder3', 'dress1', 'NO'), ('coder1', 'dress2', 'YES'), ('coder2', 'dress2', 'NO')]
    annotation_task = AnnotationTask(data)
    self.assertAlmostEqual(annotation_task.alpha(), -0.3333333)