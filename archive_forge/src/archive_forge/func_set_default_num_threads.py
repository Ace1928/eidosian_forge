import os
from packaging.version import Version, parse
from ... import logging
from ..base import CommandLine, CommandLineInputSpec, traits, isdefined, PackageInfo
@classmethod
def set_default_num_threads(cls, num_threads):
    """Set the default number of threads for ITK calls

        This method is used to set the default number of ITK threads for all
        the ANTS interfaces. However, setting this will not update the output
        type for any existing instances.  For these, assign the
        <instance>.inputs.num_threads
        """
    cls._num_threads = num_threads