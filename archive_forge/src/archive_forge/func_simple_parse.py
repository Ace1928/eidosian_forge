def simple_parse(content):
    """Returns blocks, where each block is a 2-tuple (kind, text).

    :kind: one of 'heading', 'release', 'section', 'empty' or 'text'.
    :text: a str, including newlines.
    """
    blocks = content.split('\n\n')
    for block in blocks:
        if block.startswith('###'):
            yield ('heading', block)
            continue
        last_line = block.rsplit('\n', 1)[-1]
        if last_line.startswith('###'):
            yield ('release', block)
        elif last_line.startswith('***'):
            yield ('section', block)
        elif block.startswith('* '):
            yield ('bullet', block)
        elif block.strip() == '':
            yield ('empty', block)
        else:
            yield ('text', block)