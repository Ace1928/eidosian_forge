import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Callable, Dict, Optional, Union, Any
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger: logging.Logger = logging.getLogger(__name__)


class DynamicActivation(nn.Module):
    """
    A custom PyTorch module that dynamically selects and applies an activation function, with detailed logging and error handling.
    """

    def __init__(self, activation_func: Callable[[torch.Tensor], torch.Tensor]) -> None:
        """
        Initialize the DynamicActivation module with a specific activation function.

        Parameters:
            activation_func (Callable[[torch.Tensor], torch.Tensor]): The activation function to be used in the forward pass.
        """
        super(DynamicActivation, self).__init__()
        self.activation_func: Callable[[torch.Tensor], torch.Tensor] = activation_func
        logger.debug(
            f"DynamicActivation initialized with activation function: {self.activation_func}"
        )

    def forward(
        self,
        x: torch.Tensor,
        batch_id: Optional[int] = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Perform the forward pass using the specified activation function, with detailed logging and error handling.

        Parameters:
            x (torch.Tensor): The input tensor to be processed.
            batch_id (Optional[int]): An optional batch identifier for logging purposes.
            deterministic (bool): A flag to determine if the operation should be deterministic.

        Returns:
            torch.Tensor: The activated tensor after applying the activation function.

        Raises:
            RuntimeError: If any exception occurs during the forward pass.
        """
        logger.debug(
            f"Forward pass initiated for batch_id={batch_id}, deterministic={deterministic}. Input tensor: {x}"
        )
        try:
            output: torch.Tensor = self.activation_func(x)
            logger.debug(f"Output tensor after activation: {output}")
        except Exception as e:
            logger.error(
                f"An error occurred during the forward pass for batch_id={batch_id}: {e}",
                exc_info=True,
            )
            raise RuntimeError(f"Forward pass failed due to: {e}") from e
        return output


class ActivationDictionary:
    """
    A class meticulously designed to encapsulate a comprehensive dictionary of activation functions.
    This class provides a structured and detailed approach to accessing various activation functions
    through lambda expressions, facilitating dynamic selection and application within
    neural network architectures, with an emphasis on exhaustive type annotations and detailed logging.

    Attributes:
        activation_types (Dict[str, DynamicActivation]): A dictionary mapping activation function names to their
                                 corresponding lambda expressions, enabling dynamic invocation with detailed type annotations.
    """

    def __init__(self) -> None:
        """
        Initializes the ActivationDictionary with a predefined set of activation functions,
        each represented as a lambda expression for dynamic invocation, with detailed type annotations and logging.
        """
        super().__init__()
        self.activation_types: Dict[str, DynamicActivation] = {
            "ReLU": DynamicActivation(lambda x: F.relu(x)),
            "Sigmoid": DynamicActivation(lambda x: torch.sigmoid(x)),
            "Tanh": DynamicActivation(lambda x: torch.tanh(x)),
            "Softmax": DynamicActivation(
                lambda x: F.softmax(x.unsqueeze(0), dim=0).squeeze(0)
            ),
            "Linear": DynamicActivation(lambda x: x),
            "ELU": DynamicActivation(lambda x: F.elu(x)),
            "Swish": DynamicActivation(lambda x: x * torch.sigmoid(x)),
            "Leaky ReLU": DynamicActivation(
                lambda x: F.leaky_relu(x, negative_slope=0.01)
            ),
            "Parametric ReLU": DynamicActivation(
                lambda x, a=0.01: F.prelu(
                    x.unsqueeze(0), torch.tensor([a], dtype=torch.float32)
                ).squeeze(0)
            ),
            "ELU-PA": DynamicActivation(lambda x, a=0.01: F.elu(x, alpha=a)),
            "GELU": DynamicActivation(lambda x: F.gelu(x)),
            "Softplus": DynamicActivation(lambda x: F.softplus(x)),
            "Softsign": DynamicActivation(lambda x: F.softsign(x)),
            "Bent Identity": DynamicActivation(
                lambda x: (torch.sqrt(x**2 + 1) - 1) / 2 + x
            ),
            "Hard Sigmoid": DynamicActivation(lambda x: F.hardsigmoid(x)),
            "Mish": DynamicActivation(lambda x: x * torch.tanh(F.softplus(x))),
            "SELU": DynamicActivation(lambda x: F.selu(x)),
            "SiLU": DynamicActivation(lambda x: x * torch.sigmoid(x)),
            "Softshrink": DynamicActivation(lambda x: F.softshrink(x)),
            "Threshold": DynamicActivation(
                lambda x, threshold=0.1, value=0: F.threshold(x, threshold, value)
            ),
            "LogSigmoid": DynamicActivation(lambda x: F.logsigmoid(x)),
            "Hardtanh": DynamicActivation(lambda x: F.hardtanh(x)),
            "ReLU6": DynamicActivation(lambda x: F.relu6(x)),
            "RReLU": DynamicActivation(lambda x: F.rrelu(x)),
            "PReLU": DynamicActivation(
                lambda x, a=0.25: F.prelu(
                    x.unsqueeze(0), torch.tensor([a], dtype=torch.float32)
                ).squeeze(0)
            ),
            "CReLU": DynamicActivation(lambda x: torch.cat((F.relu(x), F.relu(-x)))),
            "ELiSH": DynamicActivation(
                lambda x: torch.sign(x) * (F.elu(torch.abs(x)) + 1) / 2
            ),
            "Hardshrink": DynamicActivation(lambda x: F.hardshrink(x)),
            "LogSoftmax": DynamicActivation(
                lambda x: F.log_softmax(x.unsqueeze(0), dim=0).squeeze(0)
            ),
            "Softmin": DynamicActivation(
                lambda x: F.softmin(x.unsqueeze(0), dim=0).squeeze(0)
            ),
            "Tanhshrink": DynamicActivation(lambda x: F.tanhshrink(x)),
            "LReLU": DynamicActivation(lambda x: F.leaky_relu(x, negative_slope=0.05)),
            "AReLU": DynamicActivation(lambda x, a=0.1: F.rrelu(x, lower=a, upper=a)),
            "Maxout": DynamicActivation(lambda x: torch.max(x)),
        }
        logger.debug(
            f"ActivationDictionary initialized with activation types: {self.activation_types.keys()}"
        )

    def get_activation_function(self, name: str) -> Optional[DynamicActivation]:
        """
        Retrieves an activation function by name, ensuring a robust and detailed logging of the retrieval process.

        Parameters:
            name (str): The name of the activation function to retrieve, which must be a string representing the key in the activation_types dictionary.

        Returns:
            Optional[DynamicActivation]: The activation function as a lambda expression if found, otherwise None. The return type is meticulously annotated to ensure clarity in expected output.

        Raises:
            KeyError: If the name provided does not correspond to any existing activation function, a KeyError is raised with a detailed error message.
        """
        logger.debug(f"Attempting to retrieve activation function for the name: {name}")
        try:
            activation_function: Optional[DynamicActivation] = self.activation_types[
                name
            ]
            if activation_function is not None:
                logger.debug(f"Activation function '{name}' retrieved successfully.")
            else:
                logger.debug(
                    f"Activation function '{name}' not found in the dictionary."
                )
            return activation_function
        except KeyError as e:
            logger.error(
                f"Failed to retrieve activation function for the name '{name}': {str(e)}",
                exc_info=True,
            )
            raise KeyError(
                f"No activation function found for the specified name '{name}'. Please ensure the name is correct and try again."
            ) from e
