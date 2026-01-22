import sys
import os
@property
def user_config_dir(self):
    return user_config_dir(self.appname, self.appauthor, version=self.version, roaming=self.roaming)