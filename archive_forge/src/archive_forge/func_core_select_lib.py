import os
import sysconfig
import sys
import traceback
import tempfile
import subprocess
import importlib
import kivy
from kivy.logger import Logger
def core_select_lib(category, llist, create_instance=False, base='kivy.core', basemodule=None):
    if 'KIVY_DOC' in os.environ:
        return
    category = category.lower()
    basemodule = basemodule or category
    libs_ignored = []
    errs = []
    for option, modulename, classname in llist:
        try:
            try:
                if option not in kivy.kivy_options[category]:
                    libs_ignored.append(modulename)
                    Logger.debug('{0}: Provider <{1}> ignored by config'.format(category.capitalize(), option))
                    continue
            except KeyError:
                pass
            mod = importlib.__import__(name='{2}.{0}.{1}'.format(basemodule, modulename, base), globals=globals(), locals=locals(), fromlist=[modulename], level=0)
            cls = mod.__getattribute__(classname)
            Logger.info('{0}: Provider: {1}{2}'.format(category.capitalize(), option, '({0} ignored)'.format(libs_ignored) if libs_ignored else ''))
            if create_instance:
                cls = cls()
            return cls
        except ImportError as e:
            errs.append((option, e, sys.exc_info()[2]))
            libs_ignored.append(modulename)
            Logger.debug('{0}: Ignored <{1}> (import error)'.format(category.capitalize(), option))
            Logger.trace('', exc_info=e)
        except CoreCriticalException as e:
            errs.append((option, e, sys.exc_info()[2]))
            Logger.error('{0}: Unable to use {1}'.format(category.capitalize(), option))
            Logger.error('{0}: The module raised an important error: {1!r}'.format(category.capitalize(), e.message))
            raise
        except Exception as e:
            errs.append((option, e, sys.exc_info()[2]))
            libs_ignored.append(modulename)
            Logger.trace('{0}: Unable to use {1}'.format(category.capitalize(), option))
            Logger.trace('', exc_info=e)
    err = '\n'.join(['{} - {}: {}\n{}'.format(opt, e.__class__.__name__, e, ''.join(traceback.format_tb(tb))) for opt, e, tb in errs])
    Logger.critical('{0}: Unable to find any valuable {0} provider. Please enable debug logging (e.g. add -d if running from the command line, or change the log level in the config) and re-run your app to identify potential causes\n{1}'.format(category.capitalize(), err))