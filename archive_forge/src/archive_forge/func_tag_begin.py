from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def tag_begin(self, tag_name, attributes=None):
    """Marks the beginning of the ``tag_name`` structure.

        Call :meth:`tag_end` with the same ``tag_name`` to mark the end of the
        structure.

        The attributes string is of the form "key1=value2 key2=value2 ...".
        Values may be boolean (true/false or 1/0), integer, float, string, or
        an array.

        String values are enclosed in single quotes ('). Single quotes and
        backslashes inside the string should be escaped with a backslash.

        Boolean values may be set to true by only specifying the key. eg the
        attribute string "key" is the equivalent to "key=true".

        Arrays are enclosed in '[]'. eg "rect=[1.2 4.3 2.0 3.0]".

        If no attributes are required, ``attributes`` can be omitted, an empty
        string or None.

        See cairo's Tags and Links Description for the list of tags and
        attributes.

        Invalid nesting of tags or invalid attributes will cause the context to
        shutdown with a status of ``CAIRO_STATUS_TAG_ERROR``.

        See :meth:`tag_end`.

        :param tag_name: tag name
        :param attributes: tag attributes

        *New in cairo 1.16.*

        *New in cairocffi 0.9.*

        """
    if attributes is None:
        attributes = ''
    cairo.cairo_tag_begin(self._pointer, _encode_string(tag_name), _encode_string(attributes))
    self._check_status()