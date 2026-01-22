import os
import nipype.interfaces.spm.model as spm
import nipype.interfaces.matlab as mlab
def test_onesamplettestdesign():
    assert spm.OneSampleTTestDesign._jobtype == 'stats'
    assert spm.OneSampleTTestDesign._jobname == 'factorial_design'