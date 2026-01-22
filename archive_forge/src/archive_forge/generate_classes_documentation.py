import xml.dom.minidom
import subprocess
import os
from shutil import rmtree
import keyword
from ..base import (CommandLine, CommandLineInputSpec, SEMLikeCommandLine, TraitedSpec,
import os\n\n\n"""
modules_list contains all the SEM compliant tools that should have wrappers created for them.
    launcher containtains the command line prefix wrapper arguments needed to prepare
    a proper environment for each of the modules.
    