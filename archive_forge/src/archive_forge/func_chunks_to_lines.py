def chunks_to_lines(chunks):
    """Re-split chunks into simple lines.

    Each entry in the result should contain a single newline at the end. Except
    for the last entry which may not have a final newline. If chunks is already
    a simple list of lines, we return it directly.

    :param chunks: An list/tuple of strings. If chunks is already a list of
        lines, then we will return it as-is.
    :return: A list of strings.
    """
    last_no_newline = False
    for chunk in chunks:
        if last_no_newline:
            break
        if not chunk:
            break
        elif b'\n' in chunk[:-1]:
            break
        elif chunk[-1:] != b'\n':
            last_no_newline = True
    else:
        return chunks
    from breezy import osutils
    return osutils._split_lines(b''.join(chunks))