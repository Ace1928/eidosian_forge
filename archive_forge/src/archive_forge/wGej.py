import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Callable

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ActivationFunctionRegistry:
    """Registry for activation functions to facilitate dynamic selection and learning."""

    def __init__(self):
        self.activation_functions = {
            "relu": F.relu,
            "sigmoid": torch.sigmoid,
            "tanh": torch.tanh,
            "spiking": self.spiking_activation,
            "bspline": self.b_spline_activation,
            # Additional activations can be added here
        }

    def get_activation(self, name: str) -> Callable:
        if name in self.activation_functions:
            return self.activation_functions[name]
        else:
            raise ValueError(f"Unsupported activation function: {name}")

    def spiking_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Example spiking activation function."""
        return (x > 0).float()

    def b_spline_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Example B-spline activation function."""
        # Example implementation of B-spline using PyTorch operations.
        # This is a placeholder and should be replaced with a proper B-spline implementation.
        return torch.sin(x)


class NeuronTypeProcessor:
    """Processor for neuron type specific transformations."""

    def __init__(self):
        self.neuron_types = {
            "standard": lambda x: x,
            "spiking": self.spiking_neuron,
            "graph": self.graph_neuron,
            # Additional neuron types can be added here
        }

    def process(self, x: torch.Tensor, neuron_type: str) -> torch.Tensor:
        if neuron_type in self.neuron_types:
            return self.neuron_types[neuron_type](x)
        else:
            raise ValueError(f"Unsupported neuron type: {neuron_type}")

    def spiking_neuron(self, x: torch.Tensor) -> torch.Tensor:
        """Example spiking neuron processing."""
        return x * torch.exp(-x)

    def graph_neuron(self, x: torch.Tensor) -> torch.Tensor:
        """Example graph neuron processing (e.g., message passing)."""
        return torch.mean(x, dim=0, keepdim=True)


class KolmogorovArnoldNeuron(nn.Module):
    """
    Kolmogorov-Arnold Neurons (KANs) class.
    This class defines a universal neural network module that can apply various transformations and activation functions.
    The output can be scaled by a learnable parameter.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: str = "relu",
        neuron_type: str = "standard",
        scale_output: bool = True,
    ):
        """
        Initialize the KolmogorovArnoldNeuron module.

        Parameters:
        input_size (int): The size of the input features.
        output_size (int): The size of the output features.
        activation (str): The activation function to use.
        neuron_type (str): The type of neuron.
        scale_output (bool): Whether to scale the output by a learnable parameter.
        """
        super().__init__()
        self.basis_function = nn.Linear(input_size, output_size)
        self.param = (
            nn.Parameter(torch.randn(output_size, dtype=torch.float))
            if scale_output
            else None
        )
        self.activation_registry = ActivationFunctionRegistry()
        self.neuron_processor = NeuronTypeProcessor()
        self.activation = activation
        self.neuron_type = neuron_type
        logger.info(
            f"KolmogorovArnoldNeuron initialized with input_size={input_size}, output_size={output_size}, activation={activation}, neuron_type={neuron_type}, scale_output={scale_output}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the KolmogorovArnoldNeuron module.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor after applying the linear transformation, activation function, and scaling.
        """
        try:
            x = x.float()  # Ensure input is float
            logger.debug(f"Input tensor converted to float: {x}")
            output = self.basis_function(x)

            # Apply the specified activation function
            output = self.apply_activation(output)

            # Apply neuron type specific processing
            output = self.apply_neuron_type(output)

            if self.param is not None:
                output = self.param * output
            logger.debug(f"Output tensor after transformation and activation: {output}")
            return output
        except Exception as e:
            logger.error(f"Error during forward pass: {e}")
            raise

    def apply_activation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the specified activation function.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor after applying the activation function.
        """
        try:
            activation_function = self.activation_registry.get_activation(
                self.activation
            )
            return activation_function(x)
        except Exception as e:
            logger.error(f"Error applying activation function: {e}")
            raise

    def apply_neuron_type(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply neuron type specific processing.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Processed tensor.
        """
        try:
            return self.neuron_processor.process(x, self.neuron_type)
        except Exception as e:
            logger.error(f"Error applying neuron type processing: {e}")
            raise

    def extra_repr(self) -> str:
        """
        Extra representation of the module for better debugging and logging.
        """
        return f"input_size={self.basis_function.in_features}, output_size={self.basis_function.out_features}, activation={self.activation}, neuron_type={self.neuron_type}, scale_output={self.param is not None}"


class DynamicActivationSynapse(nn.Module):
    """
    Dynamic Activation Synapses (DAS) class.
    This class defines a learnable synapse (edge) between neurons, incorporating dynamic activation functions.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: str = "relu",
        scale_output: bool = True,
    ):
        """
        Initialize the DynamicActivationSynapse module.

        Parameters:
        input_size (int): The size of the input features.
        output_size (int): The size of the output features.
        activation (str): The activation function to use.
        scale_output (bool): Whether to scale the output by a learnable parameter.
        """
        super().__init__()
        self.synapse_function = nn.Linear(input_size, output_size)
        self.param = (
            nn.Parameter(torch.randn(output_size, dtype=torch.float))
            if scale_output
            else None
        )
        self.activation_registry = ActivationFunctionRegistry()
        self.activation = activation
        logger.info(
            f"DynamicActivationSynapse initialized with input_size={input_size}, output_size={output_size}, activation={activation}, scale_output={scale_output}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DynamicActivationSynapse module.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor after applying the synapse function, activation function, and scaling.
        """
        try:
            x = x.float()  # Ensure input is float
            logger.debug(f"Input tensor converted to float: {x}")
            output = self.synapse_function(x)

            # Apply the specified activation function
            output = self.apply_activation(output)

            if self.param is not None:
                output = self.param * output
            logger.debug(
                f"Output tensor after synapse transformation and activation: {output}"
            )
            return output
        except Exception as e:
            logger.error(f"Error during forward pass: {e}")
            raise

    def apply_activation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the specified activation function.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor after applying the activation function.
        """
        try:
            activation_function = self.activation_registry.get_activation(
                self.activation
            )
            return activation_function(x)
        except Exception as e:
            logger.error(f"Error applying activation function: {e}")
            raise

    def extra_repr(self) -> str:
        """
        Extra representation of the module for better debugging and logging.
        """
        return f"input_size={self.synapse_function.in_features}, output_size={self.synapse_function.out_features}, activation={self.activation}, scale_output={self.param is not None}"


# Example usage and test
if __name__ == "__main__":
    input_tensor = torch.randn(10, 5).to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    das = DynamicActivationSynapse(input_size=5, output_size=3, activation="relu")
    das.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    output_tensor = das(input_tensor)
    print(output_tensor)
