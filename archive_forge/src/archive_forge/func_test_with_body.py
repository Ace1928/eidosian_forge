from .. import errors, mail_client, osutils, tests, urlutils
def test_with_body(self):
    claws = mail_client.Claws(None)
    cmdline = claws._get_compose_commandline('jrandom@example.org', None, None, 'This is some body text')
    self.assertEqual(['--compose', 'mailto:jrandom@example.org?body=This%20is%20some%20body%20text'], cmdline)