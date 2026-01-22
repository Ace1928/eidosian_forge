import sys
import os
from os import path
from contextlib import contextmanager
def get_application_home(self, create=False):
    """ Return the application home directory path.

            Parameters
            ----------
            create: bool
                Create the corresponding directory or not.

            Note
            ----
            - This is a directory named after the current, running
              application that imported this module that applications and
              packages can safely write non-user accessible data to i.e.
              configuration information, preferences etc.  It is a
              sub-directory of self.application_data, named after the
              directory that contains the "main" python script that started
              the process.  For example, if application foo is started with
              a script named "run.py" in a directory named "foo", then the
              application home would be: <ETSConfig.application_data>/foo,
              regardless of if it was launched with "python
              <path_to_foo>/run.py" or "cd <path_to_foo>; python run.py"

            - This is useful for library modules used in apps that need to
              store state, preferences, etc. for the specific app only, and
              not for all apps which use that library module.  If the
              library module uses ETSConfig.application_home, they can
              store prefs for the app all in one place and do not need to
              know the details of where each app might reside.

            - Do not put anything in here that the user might want to
              navigate to e.g. projects, user home files etc.

            - The actual location differs between operating systems.

       """
    if self._application_home is None:
        self._application_home = path.join(self.get_application_data(create=create), self._get_application_dirname())
    return self._application_home