from ..language.location import get_location
@property
def locations(self):
    source = self.source
    if self.positions and source:
        return [get_location(source, pos) for pos in self.positions]