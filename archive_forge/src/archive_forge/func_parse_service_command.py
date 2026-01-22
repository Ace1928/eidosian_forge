import json
import os
import readline
import sys
from yaql import __version__ as version
from yaql.language.exceptions import YaqlParsingException
from yaql.language import utils
def parse_service_command(comm):
    space_index = comm.find(' ')
    if space_index == -1:
        return (comm, None)
    func_name = comm[:space_index]
    args = comm[len(func_name) + 1:]
    return (func_name, args)