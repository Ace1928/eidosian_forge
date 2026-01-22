import sys
from io import BytesIO
from typing import List, Optional
from twisted.python import _shellcomp, reflect, usage
from twisted.python.usage import CompleteFiles, CompleteList, Completer, Completions
from twisted.trial import unittest
def test_genZshFunction(self, cmdName, optionsFQPN):
    """
    Generate completion functions for given twisted command - no errors
    should be raised

    @type cmdName: C{str}
    @param cmdName: The name of the command-line utility e.g. 'twistd'

    @type optionsFQPN: C{str}
    @param optionsFQPN: The Fully Qualified Python Name of the C{Options}
        class to be tested.
    """
    outputFile = BytesIO()
    self.patch(usage.Options, '_shellCompFile', outputFile)
    try:
        o = reflect.namedAny(optionsFQPN)()
    except Exception as e:
        raise unittest.SkipTest("Couldn't import or instantiate Options class: %s" % (e,))
    try:
        o.parseOptions(['', '--_shell-completion', 'zsh:2'])
    except ImportError as e:
        raise unittest.SkipTest('ImportError calling parseOptions(): %s', (e,))
    except SystemExit:
        pass
    else:
        self.fail('SystemExit not raised')
    outputFile.seek(0)
    self.assertEqual(1, len(outputFile.read(1)))
    outputFile.seek(0)
    outputFile.truncate()
    if hasattr(o, 'subCommands'):
        for cmd, short, parser, doc in o.subCommands:
            try:
                o.parseOptions([cmd, '', '--_shell-completion', 'zsh:3'])
            except ImportError as e:
                raise unittest.SkipTest('ImportError calling parseOptions() on subcommand: %s', (e,))
            except SystemExit:
                pass
            else:
                self.fail('SystemExit not raised')
            outputFile.seek(0)
            self.assertEqual(1, len(outputFile.read(1)))
            outputFile.seek(0)
            outputFile.truncate()
    self.flushWarnings()