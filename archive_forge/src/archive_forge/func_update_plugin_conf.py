from __future__ import (absolute_import, division, print_function)
import os
import re
import shutil
import tempfile
import types
from ansible.module_utils.six.moves import configparser
def update_plugin_conf(self, plugin, enabled=True):
    plugin_conf = '/etc/yum/pluginconf.d/%s.conf' % plugin
    if os.path.isfile(plugin_conf):
        tmpfd, tmpfile = tempfile.mkstemp()
        shutil.copy2(plugin_conf, tmpfile)
        cfg = configparser.ConfigParser()
        cfg.read([tmpfile])
        if enabled:
            cfg.set('main', 'enabled', 1)
        else:
            cfg.set('main', 'enabled', 0)
        fd = open(tmpfile, 'w+')
        cfg.write(fd)
        fd.close()
        self.module.atomic_move(tmpfile, plugin_conf)