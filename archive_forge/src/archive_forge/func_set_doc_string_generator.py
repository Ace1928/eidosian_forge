from ._gi import \
def set_doc_string_generator(func):
    """Set doc string generator function

    :param callable func:
        Callable which takes a GIInfoStruct and returns documentation for it.
    """
    global _generate_doc_string_func
    _generate_doc_string_func = func