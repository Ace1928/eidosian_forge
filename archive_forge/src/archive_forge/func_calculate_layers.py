import math
from config import global_config
def calculate_layers(self):
    """
        Calculates the number of hexagons in each layer based on the fractal pattern.

        Returns:
            list: A list containing the number of hexagons in each layer.
        """
    layers = [self.base_layer_hexagons]
    current_hexagons = self.base_layer_hexagons
    while current_hexagons > 1:
        current_hexagons = math.ceil(current_hexagons / 2) + 6
        layers.append(current_hexagons)
    return layers