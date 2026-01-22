import textwrap
def print_dedented(text):
    print('\n'.join(textwrap.dedent(text).strip().split('\n')))