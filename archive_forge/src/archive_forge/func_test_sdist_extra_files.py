import glob
import os
import sys
import tarfile
import fixtures
from pbr.tests import base
def test_sdist_extra_files(self):
    """Test that the extra files are correctly added."""
    stdout, _, return_code = self.run_setup('sdist', '--formats=gztar')
    try:
        tf_path = glob.glob(os.path.join('dist', '*.tar.gz'))[0]
    except IndexError:
        assert False, 'source dist not found'
    tf = tarfile.open(tf_path)
    names = ['/'.join(p.split('/')[1:]) for p in tf.getnames()]
    self.assertIn('extra-file.txt', names)