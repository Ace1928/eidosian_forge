import builtins as builtin_mod
import sys
import io as _io
import tokenize
from traitlets.config.configurable import Configurable
from traitlets import Instance, Float
from warnings import warn
def compute_format_data(self, result):
    """Compute format data of the object to be displayed.

        The format data is a generalization of the :func:`repr` of an object.
        In the default implementation the format data is a :class:`dict` of
        key value pair where the keys are valid MIME types and the values
        are JSON'able data structure containing the raw data for that MIME
        type. It is up to frontends to determine pick a MIME to to use and
        display that data in an appropriate manner.

        This method only computes the format data for the object and should
        NOT actually print or write that to a stream.

        Parameters
        ----------
        result : object
            The Python object passed to the display hook, whose format will be
            computed.

        Returns
        -------
        (format_dict, md_dict) : dict
            format_dict is a :class:`dict` whose keys are valid MIME types and values are
            JSON'able raw data for that MIME type. It is recommended that
            all return values of this should always include the "text/plain"
            MIME type representation of the object.
            md_dict is a :class:`dict` with the same MIME type keys
            of metadata associated with each output.

        """
    return self.shell.display_formatter.format(result)