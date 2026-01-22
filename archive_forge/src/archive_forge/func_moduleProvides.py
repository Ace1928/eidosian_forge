from zope.interface.declarations import implementer
from zope.interface.interface import Attribute
from zope.interface.interface import Interface
def moduleProvides(*interfaces):
    """
        Declare interfaces provided by a module.

        This function is used in a module definition.

        The arguments are one or more interfaces or interface
        specifications (`IDeclaration` objects).

        The given interfaces (including the interfaces in the
        specifications) are used to create the module's direct-object
        interface specification.  An error will be raised if the module
        already has an interface specification.  In other words, it is
        an error to call this function more than once in a module
        definition.

        This function is provided for convenience. It provides a more
        convenient way to call `directlyProvides` for a module. For example::

          moduleImplements(I1)

        is equivalent to::

          directlyProvides(sys.modules[__name__], I1)

        .. seealso:: `zope.interface.moduleProvides`
        """