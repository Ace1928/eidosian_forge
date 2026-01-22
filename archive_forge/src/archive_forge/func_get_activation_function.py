import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Callable, Dict, Optional, Union, Any
import logging
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
    logger.debug(f'Attempting to retrieve activation function for the name: {name}')
    try:
        activation_function: Optional[DynamicActivation] = self.activation_types[name]
        if activation_function is not None:
            logger.debug(f"Activation function '{name}' retrieved successfully.")
        else:
            logger.debug(f"Activation function '{name}' not found in the dictionary.")
        return activation_function
    except KeyError as e:
        logger.error(f"Failed to retrieve activation function for the name '{name}': {str(e)}", exc_info=True)
        raise KeyError(f"No activation function found for the specified name '{name}'. Please ensure the name is correct and try again.") from e