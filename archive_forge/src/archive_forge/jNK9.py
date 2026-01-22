# neural_network.py

import numpy as np
from config import global_config
from logger import CustomLogger


class NeuralNetwork:
    """
    Represents a fractal hexagonally structured neural network.
    """

    def __init__(self, network_structure):
        """
        Initializes the neural network with the given structure.

        Parameters:
            network_structure (dict): The structure of the neural network.
        """
        self.network_structure = network_structure
        self.logger = CustomLogger("NeuralNetwork").get_logger()
        self.weights = {}
        self.initialize_weights()

    def initialize_weights(self):
        """
        Initializes the weights of the neural network based on its structure.
        """
        for layer_name, layer_info in self.network_structure.items():
            hexagons = layer_info["Hexagons"]
            # Assuming each hexagon in a layer is fully connected to the hexagons in the next layer.
            # This is a simplification, actual connections may vary based on the fractal pattern.
            if (
                layer_name != f"Layer_{len(self.network_structure)}"
            ):  # If not the last layer
                next_layer_hexagons = self.network_structure[
                    f'Layer_{int(layer_name.split("_")[1]) + 1}'
                ]["Hexagons"]
                # Initialize weights for connections between current layer and next layer
                self.weights[layer_name] = (
                    np.random.rand(hexagons, next_layer_hexagons) - 0.5
                )  # Subtract 0.5 to center weights around 0

        self.logger.debug(f"Weights initialized: {self.weights}")

    def activate(self, x, activation_function_name):
        """
        Applies the specified activation function to the input.

        Parameters:
            x (float): The input value.
            activation_function_name (str): The name of the activation function to apply.

        Returns:
            float: The output of the activation function.
        """
        activation_function = global_config.get_activation_function(
            activation_function_name
        )
        return activation_function(x)

    def forward_propagate(self, input_data):
        """
        Performs forward propagation through the neural network.

        Parameters:
            input_data (np.array): The input data to the neural network.

        Returns:
            np.array: The output of the neural network.
        """
        current_layer_output = input_data
        for layer_index, (layer_name, layer_info) in enumerate(
            self.network_structure.items(), start=1
        ):
            if (
                layer_name in self.weights
            ):  # If there are weights leading out of this layer
                current_layer_output = np.dot(
                    current_layer_output, self.weights[layer_name]
                )  # Matrix multiplication
                # Apply activation function element-wise
                vectorized_activation = np.vectorize(
                    lambda x: self.activate(x, layer_info["Activation_Function"])
                )
                current_layer_output = vectorized_activation(current_layer_output)
                self.logger.debug(f"Layer {layer_index} output: {current_layer_output}")

        return current_layer_output


# Example usage
if __name__ == "__main__":
    from fractal_generator import FractalGenerator

    fractal_generator = FractalGenerator(global_config.base_layer_hexagons)
    neural_network_structure = fractal_generator.generate_network()
    neural_network = NeuralNetwork(neural_network_structure)
    input_data = np.random.rand(
        neural_network_structure["Layer_1"]["Hexagons"]
    )  # Random input data
    output = neural_network.forward_propagate(input_data)
    print(f"Neural network output: {output}")
