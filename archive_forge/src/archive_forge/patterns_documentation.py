from fnmatch import fnmatch, fnmatchcase

    Matches from a set of paths based on acceptable patterns and
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
        ``True`` if any of the paths matches; ``False`` otherwise.

    Doctests::
        >>> pathnames = set(["/users/gorakhargosh/foobar.py", "/var/cache/pdnsd.status", "/etc/pdnsd.conf", "/usr/local/bin/python"])
        >>> match_any_paths(pathnames)
        True
        >>> match_any_paths(pathnames, case_sensitive=False)
        True
        >>> match_any_paths(pathnames, ["*.py", "*.conf"], ["*.status"], case_sensitive=True)
        True
        >>> match_any_paths(pathnames, ["*.txt"], case_sensitive=False)
        False
        >>> match_any_paths(pathnames, ["*.txt"], case_sensitive=True)
        False
    