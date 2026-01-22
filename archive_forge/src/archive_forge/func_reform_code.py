from warnings import warn
def reform_code(annotation):
    """
    Extract the code from the Numba annotation datastructure. 

    Pygments can only highlight full multi-line strings, the Numba
    annotation is list of single lines, with indentation removed.
    """
    ident_dict = annotation['python_indent']
    s = ''
    for n, l in annotation['python_lines']:
        s = s + ' ' * ident_dict[n] + l + '\n'
    return s