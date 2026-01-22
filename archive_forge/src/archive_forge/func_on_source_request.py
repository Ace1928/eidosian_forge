import itertools
import json
import linecache
import os
import platform
import sys
from functools import partial
import pydevd_file_utils
from _pydev_bundle import pydev_log
from _pydevd_bundle._debug_adapter import pydevd_base_schema, pydevd_schema
from _pydevd_bundle._debug_adapter.pydevd_schema import (
from _pydevd_bundle.pydevd_api import PyDevdAPI
from _pydevd_bundle.pydevd_breakpoints import get_exception_class, FunctionBreakpoint
from _pydevd_bundle.pydevd_comm_constants import (
from _pydevd_bundle.pydevd_filtering import ExcludeFilter
from _pydevd_bundle.pydevd_json_debug_options import _extract_debug_options, DebugOptions
from _pydevd_bundle.pydevd_net_command import NetCommand
from _pydevd_bundle.pydevd_utils import convert_dap_log_message_to_expression, ScopeRequest
from _pydevd_bundle.pydevd_constants import (PY_IMPL_NAME, DebugInfoHolder, PY_VERSION_STR,
from _pydevd_bundle.pydevd_trace_dispatch import USING_CYTHON
from _pydevd_frame_eval.pydevd_frame_eval_main import USING_FRAME_EVAL
from _pydevd_bundle.pydevd_comm import internal_get_step_in_targets_json
from _pydevd_bundle.pydevd_additional_thread_info import set_additional_thread_info
from _pydevd_bundle.pydevd_thread_lifecycle import pydevd_find_thread_by_id
def on_source_request(self, py_db, request):
    """
        :param SourceRequest request:
        """
    source_reference = request.arguments.sourceReference
    server_filename = None
    content = None
    if source_reference != 0:
        server_filename = pydevd_file_utils.get_server_filename_from_source_reference(source_reference)
        if not server_filename:
            server_filename = pydevd_file_utils.get_source_reference_filename_from_linecache(source_reference)
        if server_filename:
            try:
                with open(server_filename, 'r') as stream:
                    content = stream.read()
            except:
                pass
            if content is None:
                lines = (linecache.getline(server_filename, i) for i in itertools.count(1))
                lines = itertools.takewhile(bool, lines)
                content = ''.join(lines) or None
        if content is None:
            frame_id = pydevd_file_utils.get_frame_id_from_source_reference(source_reference)
            pydev_log.debug('Found frame id: %s for source reference: %s', frame_id, source_reference)
            if frame_id is not None:
                try:
                    content = self.api.get_decompiled_source_from_frame_id(py_db, frame_id)
                except Exception:
                    pydev_log.exception('Error getting source for frame id: %s', frame_id)
                    content = None
    body = SourceResponseBody(content or '')
    response_args = {'body': body}
    if content is None:
        if source_reference == 0:
            message = 'Source unavailable'
        elif server_filename:
            message = 'Unable to retrieve source for %s' % (server_filename,)
        else:
            message = 'Invalid sourceReference %d' % (source_reference,)
        response_args.update({'success': False, 'message': message})
    response = pydevd_base_schema.build_response(request, kwargs=response_args)
    return NetCommand(CMD_RETURN, 0, response, is_json=True)