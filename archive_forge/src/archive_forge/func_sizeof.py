from struct import Struct as Packer
from .lib.py3compat import BytesIO, advance_iterator, bchr
from .lib import Container, ListContainer, LazyContainer
def sizeof(self, context=None):
    """
        Calculate the size of this object, optionally using a context.

        Some constructs have no fixed size and can only know their size for a
        given hunk of data; these constructs will raise an error if they are
        not passed a context.

        :param ``Container`` context: contextual data

        :returns: int of the length of this construct
        :raises SizeofError: the size could not be determined
        """
    if context is None:
        context = Container()
    try:
        return self._sizeof(context)
    except Exception as e:
        raise SizeofError(e)