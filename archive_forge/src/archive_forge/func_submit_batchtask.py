from nipype.pipeline.plugins.base import SGELikeBatchManagerBase
from nipype.interfaces.utility import Function
import nipype.pipeline.engine as pe
import pytest
from unittest.mock import patch
import subprocess
def submit_batchtask(self, scriptfile, node):
    self._pending[1] = node.output_dir()
    subprocess.call(['bash', scriptfile])
    return 1