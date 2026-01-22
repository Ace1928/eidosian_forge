from a single quote by the algorithm. Therefore, a text like::
import re, sys
def test_html_tags(self):
    text = '<a src="foo">more</a>'
    self.assertEqual(smartyPants(text), text)