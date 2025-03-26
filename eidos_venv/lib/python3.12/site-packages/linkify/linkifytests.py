import unittest
from linkify import linkify

class LinkifyTestCase(unittest.TestCase):
  
  def testValidYoutubeLink(self):
    """
      Verify that link returns a valid youtube link structure
    """
    assert linkify('http://www.youtube.com/watch?v=oAPMuGCa-Ow')
    
  def testValidVimeoLink(self):
    """
      Verify that link returns a valid vimeo link structure
    """
    assert linkify('http://vimeo.com/5936810')
    
  def testInvalidLink(self):
    """
      Verify that invalid link returns an exception
    """
    assert not linkify('invalid stuff')
    
  def testLinkEmpty(self):
    """
      Verify that empty value returns an exception
    """
    assert not linkify('')

if __name__ == "__main__":
    unittest.main()