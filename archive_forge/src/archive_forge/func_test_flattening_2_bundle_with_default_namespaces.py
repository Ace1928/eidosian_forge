from prov.model import ProvDocument
def test_flattening_2_bundle_with_default_namespaces(self):
    prov_doc = document_with_n_bundles_having_default_namespace(2)
    prov_doc.set_default_namespace('http://www.example.org/default/0')
    flattened = prov_doc.flattened()
    self.do_tests(flattened)