from _pydevd_bundle.pydevd_constants import STATE_SUSPEND, JINJA2_SUSPEND
from _pydevd_bundle.pydevd_comm import CMD_SET_BREAK, CMD_ADD_EXCEPTION_BREAK
from pydevd_file_utils import canonical_normalized_path
from _pydevd_bundle.pydevd_frame_utils import add_exception_to_frame, FCode
from _pydev_bundle import pydev_log
from pydevd_plugins.pydevd_line_validation import LineBreakpointWithLazyValidation, ValidationInfo
from _pydev_bundle.pydev_override import overrides
from _pydevd_bundle.pydevd_api import PyDevdAPI
def collect_context(self, frame):
    res = {}
    for k, v in frame.f_locals.items():
        if not k.startswith('l_'):
            res[k] = v
        elif v and (not _is_missing(v)):
            res[self._get_real_var_name(k[2:])] = v
    if self.back_context is not None:
        for k, v in self.back_context.items():
            res[k] = v
    return res