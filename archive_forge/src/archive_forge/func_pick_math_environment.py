def pick_math_environment(code, numbered=False):
    """Return the right math environment to display `code`.

    The test simply looks for line-breaks (``\\``) outside environments.
    Multi-line formulae are set with ``align``, one-liners with
    ``equation``.

    If `numbered` evaluates to ``False``, the "starred" versions are used
    to suppress numbering.
    """
    chunks = code.split('\\begin{')
    toplevel_code = ''.join([chunk.split('\\end{')[-1] for chunk in chunks])
    if toplevel_code.find('\\\\') >= 0:
        env = 'align'
    else:
        env = 'equation'
    if not numbered:
        env += '*'
    return env