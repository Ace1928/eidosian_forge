from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_constants import DebugInfoHolder, IS_WINDOWS, IS_JYTHON, \
from _pydev_bundle._pydev_filesystem_encoding import getfilesystemencoding
from _pydevd_bundle.pydevd_comm_constants import file_system_encoding, filesystem_encoding_is_utf8
from _pydev_bundle.pydev_log import error_once
import json
import os.path
import sys
import itertools
import ntpath
from functools import partial
def setup_client_server_paths(paths):
    """paths is the same format as PATHS_FROM_ECLIPSE_TO_PYTHON"""
    global map_file_to_client
    global map_file_to_server
    global _last_client_server_paths_set
    global _next_source_reference
    _last_client_server_paths_set = paths[:]
    _source_reference_to_server_filename.clear()
    _client_filename_in_utf8_to_source_reference.clear()
    _next_source_reference = partial(next, itertools.count(1))
    python_sep = '\\' if IS_WINDOWS else '/'
    eclipse_sep = '\\' if _ide_os == 'WINDOWS' else '/'
    norm_filename_to_server_container = {}
    norm_filename_to_client_container = {}
    initial_paths = []
    initial_paths_with_end_sep = []
    paths_from_eclipse_to_python = []
    paths_from_eclipse_to_python_with_end_sep = []
    for i, (path0, path1) in enumerate(paths):
        force_only_slash = path0.endswith(('/', '\\')) and path1.endswith(('/', '\\'))
        if not force_only_slash:
            path0 = _fix_path(path0, eclipse_sep, False)
            path1 = _fix_path(path1, python_sep, False)
            initial_paths.append((path0, path1))
            paths_from_eclipse_to_python.append((_normcase_from_client(path0), normcase(path1)))
        path0 = _fix_path(path0, eclipse_sep, True)
        path1 = _fix_path(path1, python_sep, True)
        initial_paths_with_end_sep.append((path0, path1))
        paths_from_eclipse_to_python_with_end_sep.append((_normcase_from_client(path0), normcase(path1)))
    initial_paths = initial_paths_with_end_sep + initial_paths
    paths_from_eclipse_to_python = paths_from_eclipse_to_python_with_end_sep + paths_from_eclipse_to_python
    if not paths_from_eclipse_to_python:
        map_file_to_client = _original_file_to_client
        map_file_to_server = _original_map_file_to_server
        return

    def _map_file_to_server(filename, cache=norm_filename_to_server_container):
        try:
            return cache[filename]
        except KeyError:
            if eclipse_sep != python_sep:
                filename = filename.replace(python_sep, eclipse_sep)
            translated = filename
            translated_normalized = _normcase_from_client(filename)
            for eclipse_prefix, server_prefix in paths_from_eclipse_to_python:
                if translated_normalized.startswith(eclipse_prefix):
                    found_translation = True
                    if DEBUG_CLIENT_SERVER_TRANSLATION:
                        pydev_log.critical('pydev debugger: replacing to server: %s', filename)
                    translated = server_prefix + filename[len(eclipse_prefix):]
                    if DEBUG_CLIENT_SERVER_TRANSLATION:
                        pydev_log.critical('pydev debugger: sent to server: %s - matched prefix: %s', translated, eclipse_prefix)
                    break
            else:
                found_translation = False
            if eclipse_sep != python_sep:
                translated = translated.replace(eclipse_sep, python_sep)
            if found_translation:
                translated = absolute_path(translated)
            elif not os_path_exists(translated):
                if not translated.startswith('<'):
                    error_once('pydev debugger: unable to find translation for: "%s" in [%s] (please revise your path mappings).\n', filename, ', '.join(['"%s"' % (x[0],) for x in paths_from_eclipse_to_python]))
            else:
                translated = absolute_path(translated)
            cache[filename] = translated
            return translated

    def _map_file_to_client(filename, cache=norm_filename_to_client_container):
        try:
            return cache[filename]
        except KeyError:
            abs_path = absolute_path(filename)
            translated_proper_case = get_path_with_real_case(abs_path)
            translated_normalized = normcase(abs_path)
            path_mapping_applied = False
            if translated_normalized.lower() != translated_proper_case.lower():
                if DEBUG_CLIENT_SERVER_TRANSLATION:
                    pydev_log.critical('pydev debugger: translated_normalized changed path (from: %s to %s)', translated_proper_case, translated_normalized)
            for i, (eclipse_prefix, python_prefix) in enumerate(paths_from_eclipse_to_python):
                if translated_normalized.startswith(python_prefix):
                    if DEBUG_CLIENT_SERVER_TRANSLATION:
                        pydev_log.critical('pydev debugger: replacing to client: %s', translated_normalized)
                    eclipse_prefix = initial_paths[i][0]
                    translated = eclipse_prefix + translated_proper_case[len(python_prefix):]
                    if DEBUG_CLIENT_SERVER_TRANSLATION:
                        pydev_log.critical('pydev debugger: sent to client: %s - matched prefix: %s', translated, python_prefix)
                    path_mapping_applied = True
                    break
            else:
                if DEBUG_CLIENT_SERVER_TRANSLATION:
                    pydev_log.critical('pydev debugger: to client: unable to find matching prefix for: %s in %s', translated_normalized, [x[1] for x in paths_from_eclipse_to_python])
                translated = translated_proper_case
            if eclipse_sep != python_sep:
                translated = translated.replace(python_sep, eclipse_sep)
            translated = _path_to_expected_str(translated)
            cache[filename] = (translated, path_mapping_applied)
            if translated not in _client_filename_in_utf8_to_source_reference:
                if path_mapping_applied:
                    source_reference = 0
                else:
                    source_reference = _next_source_reference()
                    pydev_log.debug('Created source reference: %s for untranslated path: %s', source_reference, filename)
                _client_filename_in_utf8_to_source_reference[translated] = source_reference
                _source_reference_to_server_filename[source_reference] = filename
            return (translated, path_mapping_applied)
    map_file_to_server = _map_file_to_server
    map_file_to_client = _map_file_to_client