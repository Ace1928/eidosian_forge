from _pydevd_bundle.pydevd_breakpoints import LineBreakpoint
from _pydevd_bundle.pydevd_api import PyDevdAPI
import bisect
from _pydev_bundle import pydev_log
def verify_breakpoints(self, py_db, canonical_normalized_filename, template_breakpoints_for_file, template):
    """
        This function should be called whenever a rendering is detected.

        :param str canonical_normalized_filename:
        :param dict[int:LineBreakpointWithLazyValidation] template_breakpoints_for_file:
        """
    valid_lines_frozenset, sorted_lines = self._collect_valid_lines_in_template(template)
    self._canonical_normalized_filename_to_last_template_lines[canonical_normalized_filename] = (valid_lines_frozenset, sorted_lines)
    self._verify_breakpoints_with_lines_collected(py_db, canonical_normalized_filename, template_breakpoints_for_file, valid_lines_frozenset, sorted_lines)