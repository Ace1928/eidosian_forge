from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.core.framework import summary_pb2
from tensorflow.python.framework import test_util
from tensorflow.python.summary.writer import writer
from tensorflow.python.summary.writer import writer_cache
def assert_summaries(self, test_case, expected_logdir=None, expected_graph=None, expected_summaries=None, expected_added_graphs=None, expected_added_meta_graphs=None, expected_session_logs=None):
    """Assert expected items have been added to summary writer."""
    if expected_logdir is not None:
        test_case.assertEqual(expected_logdir, self._logdir)
    if expected_graph is not None:
        test_case.assertTrue(expected_graph is self._graph)
    expected_summaries = expected_summaries or {}
    for step in expected_summaries:
        test_case.assertTrue(step in self._summaries, msg='Missing step %s from %s.' % (step, self._summaries.keys()))
        actual_simple_values = {}
        for step_summary in self._summaries[step]:
            for v in step_summary.value:
                if 'global_step/sec' != v.tag:
                    actual_simple_values[v.tag] = v.simple_value
        test_case.assertEqual(expected_summaries[step], actual_simple_values)
    if expected_added_graphs is not None:
        test_case.assertEqual(expected_added_graphs, self._added_graphs)
    if expected_added_meta_graphs is not None:
        test_case.assertEqual(len(expected_added_meta_graphs), len(self._added_meta_graphs))
        for expected, actual in zip(expected_added_meta_graphs, self._added_meta_graphs):
            test_util.assert_meta_graph_protos_equal(test_case, expected, actual)
    if expected_session_logs is not None:
        test_case.assertEqual(expected_session_logs, self._added_session_logs)