import os
import os.path
import posixpath
import stat
from .compat import Collection, Iterable, string_types, unicode
def detailed_match_files(patterns, files, all_matches=None):
    """
	Matches the files to the patterns, and returns which patterns matched
	the files.

	*patterns* (:class:`~collections.abc.Iterable` of :class:`~pathspec.pattern.Pattern`)
	contains the patterns to use.

	*files* (:class:`~collections.abc.Iterable` of :class:`str`) contains
	the normalized file paths to be matched against *patterns*.

	*all_matches* (:class:`boot` or :data:`None`) is whether to return all
	matches patterns (:data:`True`), or only the last matched pattern
	(:data:`False`). Default is :data:`None` for :data:`False`.

	Returns the matched files (:class:`dict`) which maps each matched file
	(:class:`str`) to the patterns that matched in order (:class:`.MatchDetail`).
	"""
    all_files = files if isinstance(files, Collection) else list(files)
    return_files = {}
    for pattern in patterns:
        if pattern.include is not None:
            result_files = pattern.match(all_files)
            if pattern.include:
                for result_file in result_files:
                    if result_file in return_files:
                        if all_matches:
                            return_files[result_file].patterns.append(pattern)
                        else:
                            return_files[result_file].patterns[0] = pattern
                    else:
                        return_files[result_file] = MatchDetail([pattern])
            else:
                for file in result_files:
                    del return_files[file]
    return return_files