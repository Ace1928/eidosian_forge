from prov.model import *
def test_entity_0(self):
    document = self.new_document()
    a = document.entity(EX_NS['e0'])
    a.add_attributes([(EX_NS['tag2'], Literal('guten tag', langtag='de')), ('prov:Location', 'un llieu'), (PROV['Location'], 1), (PROV['Location'], 2.0), (PROV['Location'], EX_NS.uri + 'london')])
    self.do_tests(document)