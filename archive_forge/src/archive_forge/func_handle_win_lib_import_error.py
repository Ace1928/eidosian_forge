import os
import sysconfig
import sys
import traceback
import tempfile
import subprocess
import importlib
import kivy
from kivy.logger import Logger
def handle_win_lib_import_error(category, provider, mod_name):
    if sys.platform != 'win32':
        return
    assert mod_name.startswith('kivy.')
    kivy_root = os.path.dirname(kivy.__file__)
    dirs = mod_name[5:].split('.')
    mod_path = os.path.join(kivy_root, *dirs)
    if hasattr(sys, 'gettotalrefcount'):
        mod_path += '._d'
    mod_path += '.cp{}{}-{}.pyd'.format(sys.version_info.major, sys.version_info.minor, sysconfig.get_platform().replace('-', '_'))
    if not os.path.exists(mod_path):
        Logger.debug('{}: Failed trying to import "{}" for provider {}. Compiled file does not exist. Have you perhaps forgotten to compile Kivy, or did not install all required dependencies?'.format(category, provider, mod_path))
        return
    env_var = 'KIVY_{}_DEPENDENCY_WALKER'.format(provider.upper())
    if env_var not in os.environ:
        Logger.debug('{0}: Failed trying to import the "{1}" provider from "{2}". This error is often encountered when a dependency is missing, or if there are multiple copies of the same dependency dll on the Windows PATH and they are incompatible with each other. This can occur if you are mixing installations (such as different python installations, like anaconda python and a system python) or if another unrelated program added its directory to the PATH. Please examine your PATH and python installation for potential issues. To further troubleshoot a "DLL load failed" error, please download "Dependency Walker" (64 or 32 bit version - matching your python bitness) from dependencywalker.com and set the environment variable {3} to the full path of the downloaded depends.exe file and rerun your application to generate an error report'.format(category, provider, mod_path, env_var))
        return
    depends_bin = os.environ[env_var]
    if not os.path.exists(depends_bin):
        raise ValueError('"{}" provided in {} does not exist'.format(depends_bin, env_var))
    fd, temp_file = tempfile.mkstemp(suffix='.dwi', prefix='kivy_depends_{}_log_'.format(provider), dir=os.path.expanduser('~/'))
    os.close(fd)
    Logger.info('{}: Running dependency walker "{}" on "{}" to generate troubleshooting log. Please wait for it to complete'.format(category, depends_bin, mod_path))
    Logger.debug('{}: Dependency walker command is "{}"'.format(category, [depends_bin, '/c', '/od:{}'.format(temp_file), mod_path]))
    try:
        subprocess.check_output([depends_bin, '/c', '/od:{}'.format(temp_file), mod_path])
    except subprocess.CalledProcessError as exc:
        if exc.returncode >= 65536:
            Logger.error('{}: Dependency walker failed with error code "{}". No error report was generated'.format(category, exc.returncode))
            return
    Logger.info('{}: dependency walker generated "{}" containing troubleshooting information about provider {} and its failing file "{} ({})". You can open the file in dependency walker to view any potential issues and troubleshoot it yourself. To share the file with the Kivy developers and request support, please contact us at our support channels https://kivy.org/doc/master/contact.html (not on github, unless it\'s truly a bug). Make sure to provide the generated file as well as the *complete* Kivy log being printed here. Keep in mind the generated dependency walker log file contains paths to dlls on your system used by kivy or its dependencies to help troubleshoot them, and these paths may include your name in them. Please view the log file in dependency walker before sharing to ensure you are not sharing sensitive paths'.format(category, temp_file, provider, mod_name, mod_path))