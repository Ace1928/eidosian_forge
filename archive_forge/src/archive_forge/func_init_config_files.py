import os
from traitlets.config.application import Application
from IPython.core.application import (
from IPython.core.profiledir import ProfileDir
from IPython.utils.importstring import import_item
from IPython.paths import get_ipython_dir, get_ipython_package_dir
from traitlets import Unicode, Bool, Dict, observe
def init_config_files(self):
    super(ProfileCreate, self).init_config_files()
    from IPython.terminal.ipapp import TerminalIPythonApp
    apps = [TerminalIPythonApp]
    for app_path in ('ipykernel.kernelapp.IPKernelApp',):
        app = self._import_app(app_path)
        if app is not None:
            apps.append(app)
    if self.parallel:
        from ipyparallel.apps.ipcontrollerapp import IPControllerApp
        from ipyparallel.apps.ipengineapp import IPEngineApp
        from ipyparallel.apps.ipclusterapp import IPClusterStart
        apps.extend([IPControllerApp, IPEngineApp, IPClusterStart])
    for App in apps:
        app = App()
        app.config.update(self.config)
        app.log = self.log
        app.overwrite = self.overwrite
        app.copy_config_files = True
        app.ipython_dir = self.ipython_dir
        app.profile_dir = self.profile_dir
        app.init_config_files()