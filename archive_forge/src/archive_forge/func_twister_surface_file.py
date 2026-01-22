import time
from string import ascii_lowercase
from .gui import tkMessageBox
from .vertex import Vertex
from .arrow import Arrow, default_arrow_params
from .crossings import Crossing, ECrossing
from .smooth import TikZPicture
def twister_surface_file(self):
    """
        Returns a string containing the contents of a Twister surface
        file. Raises a ValueError if there are no virtual crossings.
        """
    result = '# A Twister surface file produced by PLink.\n'
    virtual_crossings = [crossing for crossing in self.Crossings if crossing.is_virtual]
    if len(virtual_crossings) == 0:
        raise ValueError('No virtual crossings present.')
    closed_components, nonclosed_components = self.arrow_components(distinguish_closed=True)

    def component_sequence(component):
        sequence = []
        for arrow in component:
            this_arrows_crossings = []
            for index, virtual_crossing in enumerate(virtual_crossings):
                if arrow == virtual_crossing.under:
                    other_arrow = virtual_crossing.over
                elif arrow == virtual_crossing.over:
                    other_arrow = virtual_crossing.under
                else:
                    continue
                sign = arrow.dx * other_arrow.dy - arrow.dy * other_arrow.dx > 0
                this_arrows_crossings.append((arrow ^ other_arrow, index, '+' if sign else '-'))
            this_arrows_crossings.sort()
            sequence += [pm + str(index) for _, index, pm in this_arrows_crossings]
        return sequence
    curves = list(ascii_lowercase) + ['%s%d' % (letter, index) for index in range((len(closed_components) + len(nonclosed_components)) // 26) for letter in ascii_lowercase]
    i = 0
    for component in closed_components:
        result += 'annulus,%s,%s,%s#\n' % (curves[i], curves[i].swapcase(), ','.join(component_sequence(component)))
        i += 1
    for component in nonclosed_components:
        result += 'rectangle,%s,%s,%s#\n' % (curves[i], curves[i].swapcase(), ','.join(component_sequence(component)))
        i += 1
    return result