import sys
import os
@property
def site_config_dir(self):
    return site_config_dir(self.appname, self.appauthor, version=self.version, multipath=self.multipath)