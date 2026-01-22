import os
from traitlets.config.application import Application
from IPython.core.application import (
from IPython.core.profiledir import ProfileDir
from IPython.utils.importstring import import_item
from IPython.paths import get_ipython_dir, get_ipython_package_dir
from traitlets import Unicode, Bool, Dict, observe
def list_profile_dirs(self):
    profiles = list_bundled_profiles()
    if profiles:
        print()
        print('Available profiles in IPython:')
        self._print_profiles(profiles)
        print()
        print('    The first request for a bundled profile will copy it')
        print('    into your IPython directory (%s),' % self.ipython_dir)
        print('    where you can customize it.')
    profiles = list_profiles_in(self.ipython_dir)
    if profiles:
        print()
        print('Available profiles in %s:' % self.ipython_dir)
        self._print_profiles(profiles)
    profiles = list_profiles_in(os.getcwd())
    if profiles:
        print()
        print('Profiles from CWD have been removed for security reason, see CVE-2022-21699:')
    print()
    print('To use any of the above profiles, start IPython with:')
    print('    ipython --profile=<name>')
    print()