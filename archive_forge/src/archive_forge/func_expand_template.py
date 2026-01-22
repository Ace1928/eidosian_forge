from ._constants import *
def expand_template(template, match):
    g = match.group
    empty = match.string[:0]
    groups, literals = template
    literals = literals[:]
    try:
        for index, group in groups:
            literals[index] = g(group) or empty
    except IndexError:
        raise error('invalid group reference %d' % index) from None
    return empty.join(literals)