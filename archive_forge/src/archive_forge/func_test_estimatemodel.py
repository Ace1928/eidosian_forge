import os
import nipype.interfaces.spm.model as spm
import nipype.interfaces.matlab as mlab
def test_estimatemodel():
    assert spm.EstimateModel._jobtype == 'stats'
    assert spm.EstimateModel._jobname == 'fmri_est'