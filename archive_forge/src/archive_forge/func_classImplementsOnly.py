from zope.interface.declarations import implementer
from zope.interface.interface import Attribute
from zope.interface.interface import Interface
def classImplementsOnly(class_, *interfaces):
    """
        Declare the only interfaces implemented by instances of a class.

        The arguments after the class are one or more interfaces or
        interface specifications (`IDeclaration` objects).

        The interfaces given (including the interfaces in the
        specifications) replace any previous declarations.

        Consider the following example::

          class C(A, B):
             ...

          classImplements(C, IA, IB. IC)
          classImplementsOnly(C. I1, I2)

        Instances of ``C`` provide only ``I1``, ``I2``, and regardless of
        whatever interfaces instances of ``A`` and ``B`` implement.

        .. seealso:: `zope.interface.classImplementsOnly`
        """