from prov.model import *
def test_agent_5(self):
    document = self.new_document()
    a = document.agent(EX_NS['ag5'])
    a.add_attributes([(PROV_LABEL, 'agent5'), (PROV_LABEL, Literal('hello')), (PROV_LABEL, Literal('bye', langtag='en')), (PROV_LABEL, Literal('bonjour', langtag='french'))])
    self.do_tests(document)