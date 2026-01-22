from twisted.trial import unittest
from twisted.words.protocols.jabber import error
from twisted.words.xish import domish
def test_textLang(self) -> None:
    """
        Test parsing of an error element with a text with a defined language.
        """
    text = self.error.addElement(('errorns', 'text'))
    text[NS_XML, 'lang'] = 'en_US'
    text.addContent('test')
    result = error._parseError(self.error, 'errorns')
    self.assertEqual('en_US', result['textLang'])