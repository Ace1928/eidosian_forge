from fnmatch import fnmatch, fnmatchcase
def filter_paths(pathnames, included_patterns=None, excluded_patterns=None, case_sensitive=True):
    """
    Filters from a set of paths based on acceptable patterns and
    ignorable patterns.

    :param pathnames:
        A list of path names that will be filtered based on matching and
        ignored patterns.
    :param included_patterns:
        Allow filenames matching wildcard patterns specified in this list.
        If no pattern list is specified, ["*"] is used as the default pattern,
        which matches all files.
    :param excluded_patterns:
        Ignores filenames matching wildcard patterns specified in this list.
        If no pattern list is specified, no files are ignored.
    :param case_sensitive:
        ``True`` if matching should be case-sensitive; ``False`` otherwise.
    :returns:
        A list of pathnames that matched the allowable patterns and passed
        through the ignored patterns.

    Doctests::
        >>> pathnames = set(["/users/gorakhargosh/foobar.py", "/var/cache/pdnsd.status", "/etc/pdnsd.conf", "/usr/local/bin/python"])
        >>> set(filter_paths(pathnames)) == pathnames
        True
        >>> set(filter_paths(pathnames, case_sensitive=False)) == pathnames
        True
        >>> set(filter_paths(pathnames, ["*.py", "*.conf"], ["*.status"], case_sensitive=True)) == set(["/users/gorakhargosh/foobar.py", "/etc/pdnsd.conf"])
        True
    """
    included = ['*'] if included_patterns is None else included_patterns
    excluded = [] if excluded_patterns is None else excluded_patterns
    for pathname in pathnames:
        if _match_path(pathname, included, excluded, case_sensitive):
            yield pathname