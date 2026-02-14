import pandas as pd
import numpy as np
import itertools
import logging
from typing import List, Tuple

# Configure logging to capture debug information for tracing computation values
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def estimate_memory_usage(
    probabilities_set: List[np.ndarray],
    H_X_set: List[float],
    H_Y_set: List[float],
    H_XY_set: List[float],
    P_set: List[float],
    E_set: List[float],
    error_detection_rate_set: List[float],
    correction_capability_set: List[float],
    adaptation_rate_set: List[float],
    spatial_scale_set: List[float],
    temporal_scale_set: List[float]
) -> Tuple[int, float]:
    """
    Estimates the memory usage for a DataFrame containing all combinations of parameters.

    Args:
        probabilities_set: List of probability distributions.
        H_X_set: List of entropies for X.
        H_Y_set: List of entropies for Y.
        H_XY_set: List of joint entropies for X and Y.
        P_set: List of values for P.
        E_set: List of values for E.
        error_detection_rate_set: List of error detection rates.
        correction_capability_set: List of correction capabilities.
        adaptation_rate_set: List of adaptation rates.
        spatial_scale_set: List of spatial scales.
        temporal_scale_set: List of temporal scales.

    Returns:
        A tuple containing the total number of combinations and the estimated memory usage in GB.
    """
    try:
        # Generate all combinations
        all_combinations = list(itertools.product(
            probabilities_set, H_X_set, H_Y_set, H_XY_set, P_set, E_set,
            error_detection_rate_set, correction_capability_set, adaptation_rate_set,
            spatial_scale_set, temporal_scale_set
        ))

        # Calculate the total number of combinations
        total_combinations = len(all_combinations)
        logging.debug(f"Total number of combinations: {total_combinations}")

        # Estimating memory usage
        float_size = 8  # bytes for a double precision float
        overhead_per_float = 16  # estimated bytes for DataFrame overhead per float
        entry_size = float_size + overhead_per_float
        num_columns = 12  # Number of parameters in each combination

        memory_per_row = entry_size * num_columns
        total_memory = memory_per_row * total_combinations

        total_memory_gb = total_memory / (1024 ** 3)  # Convert bytes to GB
        logging.debug(f"Estimated memory usage in GB: {total_memory_gb}")

        return total_combinations, total_memory_gb
    except Exception as e:
        logging.error(f"Error estimating memory usage: {e}")
        raise

# Example parameters, as mentioned
probabilities_set = [np.array([p1, p2, 1 - p1 - p2]) for p1 in np.linspace(0.1, 0.9, 9) for p2 in np.linspace(0.1, 0.9, 9) if p1 + p2 <= 1]
H_X_set = np.linspace(0.2, 2.0, 5).tolist()
H_Y_set = np.linspace(0.2, 2.0, 5).tolist()
H_XY_set = np.linspace(0.5, 3.0, 5).tolist()
P_set = np.linspace(100.0, 2000.0, 5).tolist()
E_set = np.linspace(20.0, 100.0, 5).tolist()
error_detection_rate_set = np.linspace(0.5, 1.0, 3).tolist()
correction_capability_set = np.linspace(0.5, 1.0, 3).tolist()
adaptation_rate_set = np.linspace(0.3, 1.0, 4).tolist()
spatial_scale_set = np.linspace(0.5, 1.5, 3).tolist()
temporal_scale_set = np.linspace(0.5, 1.5, 3).tolist()

# Calculate the total number of combinations and estimated memory usage
total_combinations, total_memory_gb = estimate_memory_usage(
    probabilities_set, H_X_set, H_Y_set, H_XY_set, P_set, E_set,
    error_detection_rate_set, correction_capability_set, adaptation_rate_set,
    spatial_scale_set, temporal_scale_set
)

# Print the results
print(f"Total combinations: {total_combinations}")
print(f"Estimated memory usage in GB: {total_memory_gb}")

