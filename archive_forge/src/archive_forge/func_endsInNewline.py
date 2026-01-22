def endsInNewline(s):
    """
    Returns C{True} if this string ends in a newline.
    """
    return s[-len('\n'):] == '\n'