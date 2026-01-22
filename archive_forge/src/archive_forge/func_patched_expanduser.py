from google.appengine.ext import vendor
import os.path
def patched_expanduser(path):
    return path