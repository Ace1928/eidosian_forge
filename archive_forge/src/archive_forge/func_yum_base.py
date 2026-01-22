from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.respawn import has_respawned, respawn_module
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.yumdnf import YumDnf, yumdnf_argument_spec
import errno
import os
import re
import sys
import tempfile
from contextlib import contextmanager
from ansible.module_utils.urls import fetch_file
@property
def yum_base(self):
    if self._yum_base:
        return self._yum_base
    else:
        self._yum_base = yum.YumBase()
        self._yum_base.preconf.debuglevel = 0
        self._yum_base.preconf.errorlevel = 0
        self._yum_base.preconf.plugins = True
        self._yum_base.preconf.enabled_plugins = self.enable_plugin
        self._yum_base.preconf.disabled_plugins = self.disable_plugin
        if self.releasever:
            self._yum_base.preconf.releasever = self.releasever
        if self.installroot != '/':
            self._yum_base.preconf.root = self.installroot
            self._yum_base.conf.installroot = self.installroot
        if self.conf_file and os.path.exists(self.conf_file):
            self._yum_base.preconf.fn = self.conf_file
        if os.geteuid() != 0:
            if hasattr(self._yum_base, 'setCacheDir'):
                self._yum_base.setCacheDir()
            else:
                cachedir = yum.misc.getCacheDir()
                self._yum_base.repos.setCacheDir(cachedir)
                self._yum_base.conf.cache = 0
        if self.disable_excludes:
            self._yum_base.conf.disable_excludes = self.disable_excludes
        self._yum_base.conf.sslverify = self.sslverify
        self.yum_base.conf
        try:
            for rid in self.disablerepo:
                self.yum_base.repos.disableRepo(rid)
            self._enablerepos_with_error_checking()
        except Exception as e:
            self.module.fail_json(msg='Failure talking to yum: %s' % to_native(e))
    return self._yum_base