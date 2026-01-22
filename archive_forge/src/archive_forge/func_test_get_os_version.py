import os
import platform
import re
import shutil
import tempfile
import subprocess
import pyomo.common.unittest as unittest
import pyomo.common.envvar as envvar
from pyomo.common import DeveloperError
from pyomo.common.fileutils import this_file
from pyomo.common.download import FileDownloader, distro_available
from pyomo.common.tee import capture_output
def test_get_os_version(self):
    f = FileDownloader()
    _os, _ver = f.get_os_version(normalize=False)
    _norm = f.get_os_version(normalize=True)
    _sys = f.get_sysinfo()[0]
    if _sys == 'linux':
        dist, dist_ver = re.match('^([^0-9]+)(.*)', _norm).groups()
        self.assertNotIn('.', dist_ver)
        self.assertGreater(int(dist_ver), 0)
        if dist == 'ubuntu':
            self.assertEqual(dist_ver, ''.join(_ver.split('.')[:2]))
        else:
            self.assertEqual(dist_ver, _ver.split('.')[0])
        if distro_available:
            d, v = f._get_distver_from_distro()
            self.assertEqual(_os, d)
            self.assertEqual(_ver, v)
            self.assertTrue(v.replace('.', '').startswith(dist_ver))
        if os.path.exists('/etc/redhat-release'):
            d, v = f._get_distver_from_redhat_release()
            self.assertEqual(_os, d)
            self.assertEqual(_ver, v)
            self.assertTrue(v.replace('.', '').startswith(dist_ver))
        if subprocess.run(['lsb_release'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
            d, v = f._get_distver_from_lsb_release()
            self.assertEqual(_os, d)
            self.assertEqual(_ver, v)
            self.assertTrue(v.replace('.', '').startswith(dist_ver))
        if os.path.exists('/etc/os-release'):
            d, v = f._get_distver_from_os_release()
            self.assertEqual(_os, d)
            self.assertTrue(_ver.startswith(v))
            self.assertTrue(v.replace('.', '').startswith(dist_ver))
    elif _sys == 'darwin':
        dist, dist_ver = re.match('^([^0-9]+)(.*)', _norm).groups()
        self.assertEqual(_os, 'macos')
        self.assertEqual(dist, 'macos')
        self.assertNotIn('.', dist_ver)
        self.assertGreater(int(dist_ver), 0)
        self.assertEqual(_norm, _os + ''.join(_ver.split('.')[:2]))
    elif _sys == 'windows':
        self.assertEqual(_os, 'win')
        self.assertEqual(_norm, _os + ''.join(_ver.split('.')[:2]))
    else:
        self.assertEqual(ans, '')
    self.assertEqual((_os, _ver), FileDownloader._os_version)
    try:
        FileDownloader._os_version, tmp = (('test', '2'), FileDownloader._os_version)
        self.assertEqual(f.get_os_version(False), ('test', '2'))
        self.assertEqual(f.get_os_version(), 'test2')
    finally:
        FileDownloader._os_version = tmp