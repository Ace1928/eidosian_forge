from Bio.Blast import Record
import xml.sax
from xml.sax.handler import ContentHandler
def set_hit_id(self):
    """Record the identifier of the database sequence (PRIVATE)."""
    self._hit.hit_id = self._value
    self._hit.title = self._value + ' '