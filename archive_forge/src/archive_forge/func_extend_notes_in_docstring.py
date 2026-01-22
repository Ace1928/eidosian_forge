import sys
def extend_notes_in_docstring(cls, notes):
    """
    This decorator replaces the decorated function's docstring
    with the docstring from corresponding method in `cls`.
    It extends the 'Notes' section of that docstring to include
    the given `notes`.
    """

    def _doc(func):
        cls_docstring = getattr(cls, func.__name__).__doc__
        if cls_docstring is None:
            return func
        end_of_notes = cls_docstring.find('        References\n')
        if end_of_notes == -1:
            end_of_notes = cls_docstring.find('        Examples\n')
            if end_of_notes == -1:
                end_of_notes = len(cls_docstring)
        func.__doc__ = cls_docstring[:end_of_notes] + notes + cls_docstring[end_of_notes:]
        return func
    return _doc