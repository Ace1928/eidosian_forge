from prov import Error
@staticmethod
def load_serializers():
    """Loads all available serializers into the registry."""
    from prov.serializers.provjson import ProvJSONSerializer
    from prov.serializers.provn import ProvNSerializer
    from prov.serializers.provxml import ProvXMLSerializer
    from prov.serializers.provrdf import ProvRDFSerializer
    Registry.serializers = {'json': ProvJSONSerializer, 'rdf': ProvRDFSerializer, 'provn': ProvNSerializer, 'xml': ProvXMLSerializer}