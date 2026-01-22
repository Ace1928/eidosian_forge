import re
from . import utilities
def get_manifold(text=text):
    triangulation_text = utilities.join_long_lines(find_unique_section(text, 'TRIANGULATION'))
    if triangulation_text[:15] == '% Triangulation':
        from snappy import Manifold
        return Manifold(triangulation_text)
    if '<?xml' in triangulation_text and '<reginadata' in triangulation_text and ('<packet' in triangulation_text):
        from reginaWrapper import NTriangulationForPtolemy
        return NTriangulationForPtolemy.from_xml(triangulation_text)
    raise Exception('Triangulation format not supported: %s...' % triangulation_text[:20])