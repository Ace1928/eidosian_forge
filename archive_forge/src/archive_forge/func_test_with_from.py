from .. import errors, mail_client, osutils, tests, urlutils
def test_with_from(self):
    claws = mail_client.Claws(None)
    cmdline = claws._get_compose_commandline('jrandom@example.org', None, None, None, 'qrandom@example.com')
    self.assertEqual(['--compose', 'mailto:jrandom@example.org?from=qrandom%40example.com'], cmdline)