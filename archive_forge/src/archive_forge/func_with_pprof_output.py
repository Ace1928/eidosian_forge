import copy
from tensorflow.python.profiler import tfprof_logger
from tensorflow.python.util.tf_export import tf_export
def with_pprof_output(self, pprof_file):
    """Generate a pprof profile gzip file.

    To use the pprof file:
      pprof -png --nodecount=100 --sample_index=1 <pprof_file>

    Args:
      pprof_file: filename for output, usually suffixed with .pb.gz.
    Returns:
      self.
    """
    self._options['output'] = 'pprof:outfile=%s' % pprof_file
    return self