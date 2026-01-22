import os
import coverage
from kivy.lang.parser import Parser
def file_tracer(self, filename):
    if filename.endswith('.kv'):
        return KivyFileTracer(filename=filename)
    return None