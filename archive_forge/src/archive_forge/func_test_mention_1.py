from prov.model import *
def test_mention_1(self):
    document = self.new_document()
    document.mention(EX_NS['e2'], EX_NS['e1'], None)
    self.do_tests(document)