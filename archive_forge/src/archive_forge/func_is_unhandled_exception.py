import linecache
import os.path
import re
from _pydev_bundle import pydev_log
from _pydevd_bundle import pydevd_dont_trace
from _pydevd_bundle.pydevd_constants import (RETURN_VALUES_DICT, NO_FTRACE,
from _pydevd_bundle.pydevd_frame_utils import add_exception_to_frame, just_raised, remove_exception_from_frame, ignore_exception_trace
from _pydevd_bundle.pydevd_utils import get_clsname_for_code
from pydevd_file_utils import get_abs_path_real_path_and_base_from_frame
from _pydevd_bundle.pydevd_comm_constants import constant_to_str, CMD_SET_FUNCTION_BREAK
import sys
import dis
def is_unhandled_exception(container_obj, py_db, frame, last_raise_line, raise_lines):
    if frame.f_lineno in raise_lines:
        return True
    else:
        try_except_infos = container_obj.try_except_infos
        if try_except_infos is None:
            container_obj.try_except_infos = try_except_infos = py_db.collect_try_except_info(frame.f_code)
        if not try_except_infos:
            return True
        else:
            valid_try_except_infos = []
            for try_except_info in try_except_infos:
                if try_except_info.is_line_in_try_block(last_raise_line):
                    valid_try_except_infos.append(try_except_info)
            if not valid_try_except_infos:
                return True
            else:
                for try_except_info in try_except_infos:
                    if try_except_info.is_line_in_except_block(frame.f_lineno):
                        if frame.f_lineno == try_except_info.except_line or frame.f_lineno in try_except_info.raise_lines_in_except:
                            return True
    return False