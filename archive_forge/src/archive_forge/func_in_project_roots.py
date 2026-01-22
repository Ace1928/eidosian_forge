import fnmatch
import glob
import os.path
import sys
from _pydev_bundle import pydev_log
import pydevd_file_utils
import json
from collections import namedtuple
from _pydev_bundle._pydev_saved_modules import threading
from pydevd_file_utils import normcase
from _pydevd_bundle.pydevd_constants import USER_CODE_BASENAMES_STARTING_WITH, \
from _pydevd_bundle import pydevd_constants
def in_project_roots(self, received_filename):
    """
        Note: don't call directly. Use PyDb.in_project_scope (there's no caching here and it doesn't
        handle all possibilities for knowing whether a project is actually in the scope, it
        just handles the heuristics based on the absolute_normalized_filename without the actual frame).
        """
    DEBUG = False
    if received_filename.startswith(USER_CODE_BASENAMES_STARTING_WITH):
        if DEBUG:
            pydev_log.debug('In in_project_roots - user basenames - starts with %s (%s)', received_filename, USER_CODE_BASENAMES_STARTING_WITH)
        return True
    if received_filename.startswith(LIBRARY_CODE_BASENAMES_STARTING_WITH):
        if DEBUG:
            pydev_log.debug('Not in in_project_roots - library basenames - starts with %s (%s)', received_filename, LIBRARY_CODE_BASENAMES_STARTING_WITH)
        return False
    project_roots = self._get_project_roots()
    absolute_normalized_filename = self._absolute_normalized_path(received_filename)
    absolute_normalized_filename_as_dir = absolute_normalized_filename + ('\\' if IS_WINDOWS else '/')
    found_in_project = []
    for root in project_roots:
        if root and (absolute_normalized_filename.startswith(root) or root == absolute_normalized_filename_as_dir):
            if DEBUG:
                pydev_log.debug('In project: %s (%s)', absolute_normalized_filename, root)
            found_in_project.append(root)
    found_in_library = []
    library_roots = self._get_library_roots()
    for root in library_roots:
        if root and (absolute_normalized_filename.startswith(root) or root == absolute_normalized_filename_as_dir):
            found_in_library.append(root)
            if DEBUG:
                pydev_log.debug('In library: %s (%s)', absolute_normalized_filename, root)
        elif DEBUG:
            pydev_log.debug('Not in library: %s (%s)', absolute_normalized_filename, root)
    if not project_roots:
        in_project = not found_in_library
        if DEBUG:
            pydev_log.debug('Final in project (no project roots): %s (%s)', absolute_normalized_filename, in_project)
    else:
        in_project = False
        if found_in_project:
            if not found_in_library:
                if DEBUG:
                    pydev_log.debug('Final in project (in_project and not found_in_library): %s (True)', absolute_normalized_filename)
                in_project = True
            else:
                if max((len(x) for x in found_in_project)) > max((len(x) for x in found_in_library)):
                    in_project = True
                if DEBUG:
                    pydev_log.debug('Final in project (found in both): %s (%s)', absolute_normalized_filename, in_project)
    return in_project