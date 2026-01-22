import sys
import os
from os import path
from contextlib import contextmanager
def get_application_data(self, create=False):
    """ Return the application data directory path.

            Parameters
            ----------
            create: bool
                Create the corresponding directory or not.

            Notes
            -----
            - This is a directory that applications and packages can safely
              write non-user accessible data to i.e. configuration
              information, preferences etc.

            - Do not put anything in here that the user might want to
              navigate to e.g. projects, user data files etc.

            - The actual location differs between operating systems.

       """
    if self._application_data is None:
        self._application_data = self._initialize_application_data(create=create)
    return self._application_data