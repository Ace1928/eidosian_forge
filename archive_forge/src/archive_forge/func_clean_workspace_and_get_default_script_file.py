import os
import pytest
import nipype.interfaces.matlab as mlab
def clean_workspace_and_get_default_script_file():
    default_script_file = mlab.MatlabInputSpec().script_file
    if os.path.exists(default_script_file):
        os.remove(default_script_file)
    return default_script_file