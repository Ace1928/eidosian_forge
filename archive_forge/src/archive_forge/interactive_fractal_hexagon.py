from typing import List, Tuple, Dict
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define type aliases for clarity
Point3D = Tuple[float, float, float]
Hexagon3D = List[Point3D]

def generate_hexagon(center: Point3D, side_length: float, elevation: float) -> Hexagon3D:
    """
    Generate the vertices of a 3D hexagon centered at `center` with a given `side_length`,
    placed at a specific elevation (z-coordinate).

    This function calculates the coordinates for each vertex of the hexagon by iterating
    through the angles that define the vertices of a hexagon, utilizing trigonometric
    functions to find the x and y coordinates based on the center position.

    Args:
        center (Point3D): The center point (x, y, z) of the hexagon.
        side_length (float): The length of each side of the hexagon.
        elevation (float): The z-coordinate for all vertices of the hexagon.

    Returns:
        Hexagon3D: List of tuples representing the vertices of the hexagon.
    """
    vertices = []
    for i in range(6):
        angle_rad = math.pi / 3 * i  # Simplified from 2*pi/6
        x = center[0] + side_length * math.cos(angle_rad)
        y = center[1] + side_length * math.sin(angle_rad)
        vertices.append((x, y, elevation))
    return vertices

def generate_3d_structure(layers: int, side_length: float) -> Dict[int, List[Hexagon3D]]:
    """
    Generate a 3D structure of stacked hexagons, where each layer above is derived from
    the central points of the hexagons in the layer below. This structure forms a pyramidal
    shape made of hexagons.

    The function iterates through each layer, starting from the base layer, generating
    hexagons positioned based on the hexagons from the previous layer, progressively
    increasing the elevation for each layer.

    Args:
        layers (int): The number of layers to generate, including the base layer.
        side_length (float): The side length of each hexagon in the structure.

    Returns:
        Dict[int, List[Hexagon3D]]: A dictionary mapping each layer index to a list of
                                    hexagons (defined by their vertices) in that layer.
    """
    structure = {}
    elevation = 0.0
    elevation_step = side_length * math.sqrt(3) / 2  # Adjusted for accurate layer spacing

    # Base layer initialization
    base_center = (0.0, 0.0, elevation)
    structure[0] = [generate_hexagon(base_center, side_length, elevation)]

    # Generate subsequent layers
    for layer in range(1, layers):
        elevation += elevation_step
        previous_layer_hexagons = structure[layer - 1]
        current_layer_hexagons = []

        for hexagon in previous_layer_hexagons:
            # Generate surrounding hexagons for each hexagon in the previous layer
            for i in range(6):
                angle_rad = math.pi / 3 * i
                x = hexagon[0][0] + side_length * 2 * math.cos(angle_rad)  # Use the first point as reference
                y = hexagon[0][1] + side_length * 2 * math.sin(angle_rad)
                new_hexagon_center = (x, y, elevation)
                # Prevent duplication of hexagons in the current layer
                if not any(np.allclose(new_hexagon_center, hex[0]) for hex in current_layer_hexagons):
                    current_layer_hexagons.append(generate_hexagon(new_hexagon_center, side_length, elevation))

        structure[layer] = current_layer_hexagons

    return structure

def plot_3d_structure(structure: Dict[int, List[Hexagon3D]]):
    """
    Plot the 3D structure of hexagons using matplotlib to create an interactive 3D plot.
    This visualization helps in understanding the spatial arrangement and layering of
    the hexagons in the 3D structure.

    Args:
        structure (Dict[int, List[Hexagon3D]]): The 3D hexagonal structure to plot, 
                                                where each key is a layer index.

    Returns:
        None. This function directly displays the plot.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Iterate through each layer and hexagon, plotting their vertices
    for layer, hexagons in structure.items():
        for hexagon in hexagons:
            xs, ys, zs = zip(*hexagon + [hexagon[0]])  # Close the hexagon loop
            ax.plot(xs, ys, zs, 'b-')

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    plt.show()

# Main execution
if __name__ == "__main__":
    layers = 5  # Total layers including the base
    side_length = 1.0  # Side length of each hexagon

    structure = generate_3d_structure(layers, side_length)
    plot_3d_structure(structure)

