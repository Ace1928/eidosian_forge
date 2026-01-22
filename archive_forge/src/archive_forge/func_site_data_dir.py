import sys
import os
@property
def site_data_dir(self):
    return site_data_dir(self.appname, self.appauthor, version=self.version, multipath=self.multipath)