from . import errors, registry, urlutils
def encode_fixes_bug_urls(bug_urls):
    """Get the revision property value for a commit that fixes bugs.

    :param bug_urls: An iterable of (escaped URL, tag) tuples. These normally
        come from `get_bug_url`.
    :return: A string that will be set as the 'bugs' property of a revision
        as part of a commit.
    """
    lines = []
    for url, tag in bug_urls:
        if ' ' in url:
            raise InvalidBugUrl(url)
        lines.append('{} {}'.format(url, tag))
    return '\n'.join(lines)