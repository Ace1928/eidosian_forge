from prov.model import *
def test_membership_1(self):
    document = self.new_document()
    document.membership(EX_NS['c'], EX_NS['e1'])
    self.do_tests(document)