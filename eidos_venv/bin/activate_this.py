"""
Activate virtualenv for current interpreter:

Use exec(open(this_file).read(), {'__file__': this_file}).

This can be used when you must use an existing Python interpreter, not the virtualenv bin/python.
"""
import os
import site
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Avoid modification if the environment is already activated
if os.environ.get("VIRTUAL_ENV") == BASE_DIR:
    return

prev_sys_path = list(sys.path)

site.addsitedir(os.path.join(BASE_DIR, "lib/python{}.{}/site-packages".format(sys.version_info.major, sys.version_info.minor)))

# Adding the app's directory to the PYTHONPATH
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.dirname(BASE_DIR))

# Set the new environment variables
os.environ["VIRTUAL_ENV"] = BASE_DIR

# The new PATH with the virtualenv's bin directory first
if "PATH" in os.environ:
    os.environ["PATH"] = os.path.join(BASE_DIR, "bin") + os.pathsep + os.environ["PATH"]
else:
    os.environ["PATH"] = os.path.join(BASE_DIR, "bin")

# Add the paths for the virtualenv
sys.real_prefix = sys.prefix
sys.prefix = BASE_DIR
