from .errors import BzrError, InternalBzrError
def lazy_import(self, scope, text):
    """Convert the given text into a bunch of lazy import objects.

        This takes a text string, which should be similar to normal python
        import markup.
        """
    self._build_map(text)
    self._convert_imports(scope)