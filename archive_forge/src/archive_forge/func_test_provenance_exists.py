import os
from nibabel.optpkg import optional_package
import pytest
from nipype.utils.provenance import ProvStore, safe_encode
@needs_rdflib5
@pytest.mark.timeout(60)
def test_provenance_exists(tmpdir):
    tmpdir.chdir()
    from nipype import config
    from nipype.interfaces.base import CommandLine
    provenance_state = config.get('execution', 'write_provenance')
    hash_state = config.get('execution', 'hash_method')
    config.enable_provenance()
    CommandLine('echo hello').run()
    config.set('execution', 'write_provenance', provenance_state)
    config.set('execution', 'hash_method', hash_state)
    assert tmpdir.join('provenance.provn').check()