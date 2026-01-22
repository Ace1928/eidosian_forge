from ...commands import Command, plugin_cmds, register_command
from .. import TestCaseWithMemoryTransport
def test_strict_ascii(self):

    def bzr(*args, **kwargs):
        kwargs['encoding'] = 'ascii'
        return self.run_bzr_raw(*args, **kwargs)[0]
    register_command(cmd_echo_strict)
    try:
        self.assertEqual(b'foo', bzr('echo-strict foo'))
        self.assertRaises(UnicodeEncodeError, bzr, ['echo-strict', 'fooµ'])
    finally:
        plugin_cmds.remove('echo-strict')