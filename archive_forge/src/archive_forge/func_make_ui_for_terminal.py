import warnings
def make_ui_for_terminal(stdin, stdout, stderr):
    """Construct and return a suitable UIFactory for a text mode program.
    """
    from .text import TextUIFactory
    return TextUIFactory(stdin, stdout, stderr)