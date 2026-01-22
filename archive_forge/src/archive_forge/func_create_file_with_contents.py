import os
import tempfile
import subprocess
from subprocess import PIPE
def create_file_with_contents(file_name, data):
    with open(file_name, 'w') as file:
        file.write(data)