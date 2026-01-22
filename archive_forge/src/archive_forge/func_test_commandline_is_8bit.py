from .. import errors, mail_client, osutils, tests, urlutils
def test_commandline_is_8bit(self):
    claws = mail_client.Claws(None)
    cmdline = claws._get_compose_commandline('jrandom@example.org', 'µcosm of fun!', 'file%')
    subject_string = urlutils.quote('µcosm of fun!'.encode(osutils.get_user_encoding(), 'replace'))
    self.assertEqual(['--compose', 'mailto:jrandom@example.org?subject=%s' % subject_string, '--attach', 'file%'], cmdline)
    for item in cmdline:
        self.assertTrue(isinstance(item, str), 'Command-line item %r is not a native string!' % item)