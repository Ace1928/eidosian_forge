from . import errors, registry, urlutils
def decode_bug_urls(bug_text):
    """Decode a bug property text.

    :param bug_text: Contents of a bugs property
    :return: iterator over (url, status) tuples
    """
    for line in bug_text.splitlines():
        try:
            url, status = line.split(None, 2)
        except ValueError as exc:
            raise InvalidLineInBugsProperty(line) from exc
        if status not in ALLOWED_BUG_STATUSES:
            raise InvalidBugStatus(status)
        yield (url, status)