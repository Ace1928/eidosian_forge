def out_block(text, prefix=''):
    """Format text in blocks of 80 chars with an additional optional prefix."""
    output = ''
    for j in range(0, len(text), 80):
        output += f'{prefix}{text[j:j + 80]}\n'
    output += '\n'
    return output