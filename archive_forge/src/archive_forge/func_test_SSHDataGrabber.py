import os
import copy
import simplejson
import glob
import os.path as op
from subprocess import Popen
import hashlib
from collections import namedtuple
import pytest
import nipype
import nipype.interfaces.io as nio
from nipype.interfaces.base.traits_extension import isdefined
from nipype.interfaces.base import Undefined, TraitError
from nipype.utils.filemanip import dist_is_editable
from subprocess import check_call, CalledProcessError
@pytest.mark.skipif(no_paramiko, reason='paramiko library is not available')
@pytest.mark.skipif(no_local_ssh, reason='SSH Server is not running')
def test_SSHDataGrabber(tmpdir):
    """Test SSHDataGrabber by connecting to localhost and collecting some data."""
    old_cwd = tmpdir.chdir()
    source_dir = tmpdir.mkdir('source')
    source_hdr = source_dir.join('somedata.hdr')
    source_dat = source_dir.join('somedata.img')
    source_hdr.ensure()
    source_dat.ensure()

    def _mock_get_ssh_client(self):
        proxy = None
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect('127.0.0.1', username=os.getenv('USER'), sock=proxy, timeout=10)
        return client
    MockSSHDataGrabber = copy.copy(nio.SSHDataGrabber)
    MockSSHDataGrabber._get_ssh_client = _mock_get_ssh_client
    ssh_grabber = MockSSHDataGrabber(infields=['test'], outfields=['test_file'])
    ssh_grabber.inputs.base_directory = str(source_dir)
    ssh_grabber.inputs.hostname = '127.0.0.1'
    ssh_grabber.inputs.field_template = dict(test_file='%s.hdr')
    ssh_grabber.inputs.template = ''
    ssh_grabber.inputs.template_args = dict(test_file=[['test']])
    ssh_grabber.inputs.test = 'somedata'
    ssh_grabber.inputs.sort_filelist = True
    runtime = ssh_grabber.run()
    assert runtime.outputs.test_file == str(tmpdir.join(source_hdr.basename))
    assert tmpdir.join(source_hdr.basename).new(ext='.img').check(file=True, exists=True)
    old_cwd.chdir()