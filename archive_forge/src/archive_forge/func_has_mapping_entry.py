import bisect
from _pydevd_bundle.pydevd_constants import NULL, KeyifyList
import pydevd_file_utils
def has_mapping_entry(self, runtime_source_filename):
    """
        :param runtime_source_filename:
            Something as <ipython-cell-xxx>
        """
    key = ('has_entry', runtime_source_filename)
    try:
        return self._cache[key]
    except KeyError:
        for _absolute_normalized_filename, mapping in list(self._mappings_to_server.items()):
            for map_entry in mapping:
                if map_entry.runtime_source == runtime_source_filename:
                    self._cache[key] = True
                    return self._cache[key]
        self._cache[key] = False
        return self._cache[key]