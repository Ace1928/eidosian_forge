# config.py


class Config:
    """
    Configuration class for the fractal hexagonally structured neural network.
    """

    def __init__(self):
        # Number of hexagons on the base layer. Can be adjusted as needed.
        self.base_layer_hexagons = 3

        # Activation functions available for use in the neural network.
        # This can be expanded with more functions as needed.
        self.activation_functions = {
            "relu": lambda x: max(0, x),
            "sigmoid": lambda x: 1 / (1 + math.exp(-x)),
            "tanh": lambda x: math.tanh(x),
        }

        # Default activation function to use if none is specified.
        self.default_activation_function = "relu"

        # Verbosity level for logging. Can be adjusted between 0 (no logs) to 10 (maximum detail).
        self.verbosity_level = 10

        # Visualization options
        self.visualization_options = {
            "show_weights": True,  # Whether to display weights on the visualization
            "color_scheme": "viridis",  # Color scheme for visualization
        }

        # CSV export options
        self.csv_export_options = {
            "delimiter": ",",  # Delimiter to use in the CSV files
            "quotechar": '"',  # Character used to quote fields in the CSV files
        }

    def get_activation_function(self, name):
        """
        Retrieve an activation function by name.

        Parameters:
            name (str): The name of the activation function to retrieve.

        Returns:
            function: The activation function.
        """
        return self.activation_functions.get(
            name, self.activation_functions[self.default_activation_function]
        )


# Create a global config object that can be imported and used throughout the project.
global_config = Config()
