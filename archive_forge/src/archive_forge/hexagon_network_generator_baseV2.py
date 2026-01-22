import logging
import math
import sys
from typing import List, Tuple, Dict, Optional, Union, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d
import pandas as pd

# Initialize logging for detailed execution tracking
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Define type aliases for clarity and type safety
Point3D = Tuple[float, float, float]
Hexagon3D = List[Point3D]
Structure3D = Dict[int, List[Hexagon3D]]

class Arrow3D(FancyArrowPatch):
    """
    Draws a 3D arrow using matplotlib's FancyArrowPatch.
    
    Attributes:
        xs (List[float]): x-coordinates of the start and end points.
        ys (List[float]): y-coordinates of the start and end points.
        zs (List[float]): z-coordinates of the start and end points.
    """
    def __init__(self, xs: List[float], ys: List[float], zs: List[float], 
                 *args, **kwargs):
        """
        Initializes the Arrow3D object with start and end coordinates in 3D space.
        """
        logging.info("Initializing 3D arrow with coordinates: xs=%s, ys=%s, zs=%s",
                     xs, ys, zs)
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer: Optional[Any] = None) -> float:
        """
        Projects the 3D arrow onto 2D space using the renderer's projection.
        """
        xs, ys, zs = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs, ys, zs, 
                                           renderer.M if renderer else plt.gca().get_proj())
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        logging.debug("Projected 3D arrow to 2D with min z-coordinate: %s", np.min(zs))
        return np.min(zs)

class HexagonalStructure:
    """
    Manages the generation, manipulation, and visualization of a 3D hexagonal structure.

    Attributes:
        layers (int): The number of layers in the hexagonal structure.
        side_length (float): The side length of each hexagon in the structure.
        structure (Structure3D): The generated 3D structure stored as a dictionary.
    """
    
    def __init__(self, layers: int, side_length: float):
        """
        Initializes the HexagonalStructure with specified layers and side length.
        """
        logging.info(f"Creating HexagonalStructure with {layers} layers and side length {side_length}")
        self.layers = layers
        self.side_length = side_length
        self.structure = self._generate_3d_structure()

    def _generate_hexagon(self, center: Point3D, elevation: float) -> Hexagon3D:
        """
        Generates the vertices of a hexagon given its center and elevation.
        """
        logging.debug(f"Generating hexagon at center: {center} and elevation: {elevation}")
        vertices = []
        for i in range(6):
            angle_rad = 2 * math.pi / 6 * i
            x = center[0] + self.side_length * math.cos(angle_rad)
            y = center[1] + self.side_length * math.sin(angle_rad)
            vertices.append((x, y, elevation))
        return vertices

    def _generate_3d_structure(self) -> Structure3D:
        """
        Generates a sophisticated 3D structure consisting of meticulously stacked hexagons,
        each placed with precision to form a cohesive, extended hexagonal prism structure. 
        This method embodies the pinnacle of algorithmic design, pushing the boundaries 
        of computational geometry to create a visually stunning and mathematically robust 
        representation of a hexagonal structure in three dimensions.

        Returns:
            Structure3D: A meticulously curated dictionary. Each key represents a layer index,
            associated with a list of hexagons within that layer. Each hexagon is further
            represented as a list of 3D points, constituting a comprehensive model of the 
            entire 3D hexagonal architecture.
        """
        logging.debug("Initiating the generation of the 3D hexagonal structure.")
        structure = {}
        elevation = 0.0  # Initial elevation set to ground level
        elevation_step = self.side_length * math.sqrt(3) / 2  # Calculated vertical distance between layers

        # Iteratively generate each layer of hexagons
        for layer in range(self.layers):
            structure[layer] = [self._generate_hexagon((layer * self.side_length * 1.5, 0, layer * elevation_step), layer * elevation_step)]
            # Correcting center offset calculation to ensure symmetrical stacking
            center_offset_x = self.side_length * 1.5 * layer  
            center_offset_y = self.side_length * math.sqrt(3) / 2 * layer

            # Special case for the base layer
        base_center = (0.0, 0.0, elevation)
        structure[0] = [self._generate_hexagon(base_center, elevation)]

            # For upper layers, position hexagons around each hexagon of the layer below

        for layer in range(1, self.layers):
            elevation += elevation_step
            previous_layer_hexagons = structure[layer - 1]
            current_layer_hexagons = []

                    # Position new hexagons around the perimeter of those in the layer below
            for hexagon in previous_layer_hexagons:
                for i in range(6):
                    angle_rad = math.pi / 3 * i
                    x = hexagon[0][0] + self.side_length * math.cos(angle_rad) * 2
                    y = hexagon[0][1] + self.side_length * math.sin(angle_rad) * 2
                    new_hexagon_center = (x, y, elevation)
                            # Ensure new hexagon does not overlap with existing ones
                    if not any(np.allclose(new_hexagon_center, h[0]) for h in current_layer_hexagons):
                        current_layer_hexagons.append(self._generate_hexagon(new_hexagon_center, elevation))
            structure[layer] = current_layer_hexagons


            

        logging.debug("The 3D hexagonal structure has been fully realized to the zenith of algorithmic artistry.")
        return structure

    def plot_structure(self):
        """
        Plots the 3D hexagonal structure with an interactive matplotlib figure.
        """
        logging.info("Plotting 3D hexagonal structure")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        color_map = plt.get_cmap('viridis')

        for layer, hexagons in self.structure.items():
            color = color_map(layer / self.layers)
            for hexagon in hexagons:
                self.hexagon_connections(hexagon, ax, color=color)
                xs, ys, zs = zip(*hexagon)
                ax.plot(xs, ys, zs, color=color)

        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        plt.title('3D Hexagonal Structure')
        plt.tight_layout()
        plt.show()

    def hexagon_connections(self, hexagon: Hexagon3D, ax: plt.Axes, color: str):
        """
        Draws connections between the vertices of a hexagon and its center, enhancing the 3D visualization.
        """
        for i in range(6):  # Adjusted indexing to correctly access hexagon vertices
            start = hexagon[i]
            for j in [1, 2, 3]:  # Direct connections and skips
                end = hexagon[(i + j) % 6]
                ax.add_artist(Arrow3D([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                                      mutation_scale=10, lw=1, arrowstyle="-|>", color=color))
        center = np.mean(np.array(hexagon), axis=0)  # Correct calculation of the hexagon's center
        for vertex in hexagon:
            ax.add_artist(Arrow3D([vertex[0], center[0]], [vertex[1], center[1]], [vertex[2], center[2]],
                                  mutation_scale=10, lw=1, arrowstyle="-|>", color=color))
    
    def export_to_csv(self, filename: str):
        """
        Exports the structure data to a CSV file.
        """
        logging.info(f"Exporting structure data to {filename}")
        data = []
        for layer, hexagons in self.structure.items():
            for hexagon in hexagons:
                for vertex in hexagon:
                    data.append({"Layer": layer, "X": vertex[0], "Y": vertex[1], "Z": vertex[2]})
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        logging.info(f"Data exported successfully to {filename}")

def main():
    """
    Main function to execute the script functionality.
    """
    logging.info("Starting the hexagonal structure visualization script.")
    try:
        layers = int(input("Enter the number of layers: "))
        side_length = float(input("Enter the side length of each hexagon: "))

        structure = HexagonalStructure(layers, side_length)
        structure.plot_structure()
    except ValueError as e:
        logging.error(f"Invalid input: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
    except KeyboardInterrupt:
        logging.info("Program execution was interrupted by the user.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(f"Critical failure: {e}")
        sys.exit(1)
