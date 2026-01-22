import re
import argparse
import os
import fileinput
import logging
from xmlschema.cli import xsd_version_number, defuse_data
from xmlschema.validators import XMLSchema10, XMLSchema11
from ._observers import ObservedXMLSchema10, ObservedXMLSchema11
def get_test_program_args_parser(default_testfiles):
    """
    Gets an argument parser for building test scripts for schemas and xml files.
    The returned parser has many arguments of unittest's TestProgram plus some
    arguments for selecting testfiles and XML schema options.
    """
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('-v', '--verbose', dest='verbosity', default=1, action='store_const', const=2, help='Verbose output')
    parser.add_argument('-q', '--quiet', dest='verbosity', action='store_const', const=0, help='Quiet output')
    parser.add_argument('--locals', dest='tb_locals', action='store_true', help='Show local variables in tracebacks')
    parser.add_argument('-f', '--failfast', dest='failfast', action='store_true', help='Stop on first fail or error')
    parser.add_argument('-c', '--catch', dest='catchbreak', action='store_true', help='Catch Ctrl-C and display results so far')
    parser.add_argument('-b', '--buffer', dest='buffer', action='store_true', help='Buffer stdout and stderr during tests')
    parser.add_argument('-k', dest='patterns', action='append', default=list(), help='Only run tests which match the given substring')
    parser.add_argument('--lxml', dest='lxml', action='store_true', default=False, help='Check also with lxml.etree.XMLSchema (for XSD 1.0)')
    parser.add_argument('--codegen', action='store_true', default=False, help='Test code generation with XML data bindings module.')
    parser.add_argument('testfiles', type=str, nargs='*', default=default_testfiles, help='Test cases directory.')
    return parser