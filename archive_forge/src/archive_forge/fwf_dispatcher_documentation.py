from typing import Optional, Sequence, Tuple, Union
from modin.core.io.text.text_file_dispatcher import TextFileDispatcher

        Check support of parameters of `read_fwf` function.

        Parameters
        ----------
        filepath_or_buffer : str, path object or file-like object
            `filepath_or_buffer` parameter of `read_fwf` function.
        read_kwargs : dict
            Parameters of `read_fwf` function.
        skiprows_md : int, array or callable
            `skiprows` parameter modified for easier handling by Modin.
        header_size : int
            Number of rows that are used by header.

        Returns
        -------
        bool
            Whether passed parameters are supported or not.
        Optional[str]
            `None` if parameters are supported, otherwise an error
            message describing why parameters are not supported.
        