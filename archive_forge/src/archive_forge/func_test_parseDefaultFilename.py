from twisted.runner import inetdconf
from twisted.trial import unittest
def test_parseDefaultFilename(self) -> None:
    """
        Services are parsed from default filename.
        """
    conf = inetdconf.ServicesConf()
    conf.defaultFilename = self.servicesFilename1
    conf.parseFile()
    self.assertEqual(conf.services, {('http', 'tcp'): 80, ('http', 'udp'): 80, ('http', 'sctp'): 80, ('www', 'tcp'): 80, ('www', 'udp'): 80, ('www-http', 'tcp'): 80, ('www-http', 'udp'): 80})