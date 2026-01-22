import math
import logging


class ActivationFunctionManager:
    """
    Manages the activation functions within neural networks, ensuring a comprehensive and robust selection tailored to various network layers. This class encapsulates the complexity of activation function dynamics and provides a systematic approach to their management and application, adhering to the highest standards of software engineering and mathematical precision.

    Attributes:
        activation_types (dict): A dictionary mapping activation function names to their mathematical representations, allowing for dynamic selection and application.
        current_activation (str): The currently active activation function type, facilitating state tracking and operational consistency.

    Methods:
        initialize_activation_types(): Initializes the dictionary of activation functions with their respective mathematical implementations.
        apply_function(type: str, input: float) -> float: Applies the specified activation function to the input and returns the result, incorporating detailed logging and error handling.
    """

    def __init__(self):
        """
        Initializes the ActivationFunctionManager by setting up the foundational state and structure for managing various activation functions used within neural networks. This includes initializing the dictionary of activation functions and setting the initial state of the current activation to None.
        """
        self.initialize_activation_types()
        self.current_activation = None
        logging.info(
            "ActivationFunctionManager initialized with supported types: "
            + ", ".join(self.activation_types.keys())
        )

    def initialize_activation_types(self):
        """
        Sets up the dictionary of activation functions with their respective lambda expressions, encapsulating the mathematical logic required for each function. This method enhances modularity and maintainability of the activation function management.
        """
        self.activation_types = {
            "ReLU": lambda x: max(0, x),
            "Sigmoid": lambda x: 1 / (1 + math.exp(-x)),
            "Tanh": lambda x: math.tanh(x),
        }

    def apply_function(self, type: str, input: float) -> float:
        """
        Applies the specified activation function to the given input using advanced mathematical models. This method includes comprehensive error handling to ensure that only supported activation types are used, and it logs detailed information about the application process.

        Parameters:
            type (str): The type of activation function to apply. Must be one of the supported types defined in the activation_types dictionary.
            input (float): The input value to which the activation function will be applied.

        Returns:
            float: The output from the activation function, calculated using the appropriate mathematical model.

        Raises:
            ValueError: If the specified activation type is not supported, an error is logged and a ValueError is raised to prevent misuse of the function.
        """
        if type not in self.activation_types:
            error_message = f"Attempted to use unsupported activation type: {type}"
            logging.error(error_message)
            raise ValueError(error_message)

        self.current_activation = type
        result = self.activation_types[type](input)
        logging.debug(
            f"Applied {type} activation function to input {input}, resulting in output {result}"
        )
        return result
