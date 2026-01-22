import cupy
from cupy import _core
Return an array which satisfies the requirements.

    Args:
        a (~cupy.ndarray): The input array.
        dtype (str or dtype object, optional): The required data-type.
            If None preserve the current dtype.
        requirements (str or list of str): The requirements can be any
            of the following

            * 'F_CONTIGUOUS' ('F', 'FORTRAN') - ensure a Fortran-contiguous                 array. 
            * 'C_CONTIGUOUS' ('C', 'CONTIGUOUS') - ensure a C-contiguous array.

            * 'OWNDATA' ('O')      - ensure an array that owns its own data.

    Returns:
        ~cupy.ndarray: The input array ``a`` with specified requirements and
        type if provided.

    .. seealso:: :func:`numpy.require`

    