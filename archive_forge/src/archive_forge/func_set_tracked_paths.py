import threading
import sys
from os.path import basename
from _pydev_bundle import pydev_log
from os import scandir
import time
def set_tracked_paths(self, paths):
    """
        Note: always resets all path trackers to track the passed paths.
        """
    if not isinstance(paths, (list, tuple, set)):
        paths = (paths,)
    paths = sorted(set(paths), key=lambda path: -len(path))
    path_watchers = set()
    self._single_visit_info = _SingleVisitInfo()
    initial_time = time.time()
    for path in paths:
        sleep_time = 0.0
        path_watcher = _PathWatcher(path, self.accept_directory, self.accept_file, self._single_visit_info, max_recursion_level=self.max_recursion_level, sleep_time=sleep_time)
        path_watchers.add(path_watcher)
    actual_time = time.time() - initial_time
    pydev_log.debug('Tracking the following paths for changes: %s', paths)
    pydev_log.debug('Time to track: %.2fs', actual_time)
    pydev_log.debug('Folders found: %s', len(self._single_visit_info.visited_dirs))
    pydev_log.debug('Files found: %s', len(self._single_visit_info.file_to_mtime))
    self._path_watchers = path_watchers