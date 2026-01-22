import os
import warnings
from ..base import (
from .base import NiftySegCommand
from ..niftyreg.base import get_custom_path
from ...utils.filemanip import load_json, save_json, split_filename
def get_steps_args(self):
    if not isdefined(self.inputs.template_file):
        err = "LabelFusion requires a value for input 'template_file' when 'classifier_type' is set to 'STEPS'."
        raise NipypeInterfaceError(err)
    if not isdefined(self.inputs.kernel_size):
        err = "LabelFusion requires a value for input 'kernel_size' when 'classifier_type' is set to 'STEPS'."
        raise NipypeInterfaceError(err)
    if not isdefined(self.inputs.template_num):
        err = "LabelFusion requires a value for input 'template_num' when 'classifier_type' is set to 'STEPS'."
        raise NipypeInterfaceError(err)
    return '-STEPS %f %d %s %s' % (self.inputs.kernel_size, self.inputs.template_num, self.inputs.file_to_seg, self.inputs.template_file)