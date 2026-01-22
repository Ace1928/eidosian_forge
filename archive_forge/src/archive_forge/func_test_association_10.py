from prov.model import *
def test_association_10(self):
    document = self.new_document()
    assoc1 = document.association(EX_NS['a1'], agent=EX_NS['ag1'], identifier=EX_NS['assoc10a'])
    assoc1.add_attributes([(PROV_ROLE, 'figroll')])
    assoc2 = document.association(EX_NS['a1'], agent=EX_NS['ag2'], identifier=EX_NS['assoc10b'])
    assoc2.add_attributes([(PROV_ROLE, 'sausageroll')])
    self.do_tests(document)