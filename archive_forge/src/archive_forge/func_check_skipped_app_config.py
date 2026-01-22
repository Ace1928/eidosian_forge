import os
import warnings
import builtins
import cherrypy
def check_skipped_app_config(self):
    """Check for mounted Applications that have no config."""
    for sn, app in cherrypy.tree.apps.items():
        if not isinstance(app, cherrypy.Application):
            continue
        if not app.config:
            msg = 'The Application mounted at %r has an empty config.' % sn
            if self.global_config_contained_paths:
                msg += ' It looks like the config you passed to cherrypy.config.update() contains application-specific sections. You must explicitly pass application config via cherrypy.tree.mount(..., config=app_config)'
            warnings.warn(msg)
            return