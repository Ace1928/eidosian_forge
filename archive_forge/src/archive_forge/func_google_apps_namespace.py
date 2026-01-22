import os
import re
import warnings
from googlecloudsdk.third_party.appengine.api import lib_config
def google_apps_namespace():
    return os.environ.get(_ENV_DEFAULT_NAMESPACE, None)