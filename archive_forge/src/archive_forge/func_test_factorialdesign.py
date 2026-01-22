import os
import nipype.interfaces.spm.model as spm
import nipype.interfaces.matlab as mlab
def test_factorialdesign():
    assert spm.FactorialDesign._jobtype == 'stats'
    assert spm.FactorialDesign._jobname == 'factorial_design'