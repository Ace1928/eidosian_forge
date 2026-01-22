import os
import coverage
from kivy.lang.parser import Parser
def coverage_init(reg, options):
    reg.add_file_tracer(KivyCoveragePlugin())