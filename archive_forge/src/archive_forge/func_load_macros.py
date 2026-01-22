import sys, os
from distutils.errors import \
from distutils.ccompiler import \
from distutils import log
def load_macros(self, version):
    vsbase = 'Software\\Microsoft\\VisualStudio\\%0.1f' % version
    self.set_macro('VCInstallDir', vsbase + '\\Setup\\VC', 'productdir')
    self.set_macro('VSInstallDir', vsbase + '\\Setup\\VS', 'productdir')
    net = 'Software\\Microsoft\\.NETFramework'
    self.set_macro('FrameworkDir', net, 'installroot')
    try:
        if version > 7.0:
            self.set_macro('FrameworkSDKDir', net, 'sdkinstallrootv1.1')
        else:
            self.set_macro('FrameworkSDKDir', net, 'sdkinstallroot')
    except KeyError as exc:
        raise DistutilsPlatformError('Python was built with Visual Studio 2003;\nextensions must be built with a compiler than can generate compatible binaries.\nVisual Studio 2003 was not found on this system. If you have Cygwin installed,\nyou can try compiling with MingW32, by passing "-c mingw32" to setup.py.')
    p = 'Software\\Microsoft\\NET Framework Setup\\Product'
    for base in HKEYS:
        try:
            h = RegOpenKeyEx(base, p)
        except RegError:
            continue
        key = RegEnumKey(h, 0)
        d = read_values(base, '%s\\%s' % (p, key))
        self.macros['$(FrameworkVersion)'] = d['version']