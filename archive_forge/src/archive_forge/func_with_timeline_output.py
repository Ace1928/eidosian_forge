import copy
from tensorflow.python.profiler import tfprof_logger
from tensorflow.python.util.tf_export import tf_export
def with_timeline_output(self, timeline_file):
    """Generate a timeline json file."""
    self._options['output'] = 'timeline:outfile=%s' % timeline_file
    return self