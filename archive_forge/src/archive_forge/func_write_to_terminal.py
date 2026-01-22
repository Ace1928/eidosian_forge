import logging
import sys
import time
import uuid
import pytest
import panel as pn
def write_to_terminal(term):
    term.write('This is written directly to the terminal.\n')