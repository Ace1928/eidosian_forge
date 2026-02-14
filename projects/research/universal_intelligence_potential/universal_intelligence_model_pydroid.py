import numpy as np
import logging
from typing import Dict, Tuple, List, Any

# Set up detailed logging with timestamps and log levels
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the parameters for the model with default values
default_params: Dict[str, float] = {
    "k": 1.0,  # Scaling constant for overall intelligence calculation
    # Additional parameters with default scaling factors
    "alpha_H": 1.0, "alpha_I": 1.0, "alpha_O": 1.0, "alpha_Em": 1.0, "alpha_A": 1.0,
    "alpha_Volume": 1.0, "alpha_t": 1.0, "alpha_Pi": 1.0, "alpha_log": 1.0,
    "alpha_HX": 1.0, "alpha_HY": 1.0, "alpha_HXY": 1.0, "alpha_P": 1.0, "alpha_E": 1.0,
    "alpha_Error_Detection": 1.0, "alpha_Correction": 1.0, "alpha_Adaptation_Rate": 1.0,
    "alpha_Spatial_Scale": 1.0, "alpha_Temporal_Scale": 1.0
}

# Define functions for each component with type annotations, detailed logging, and docstrings
def shannon_entropy(probabilities: np.ndarray, params: Dict[str, float]) -> float:
    """Calculate Shannon Entropy with parameter scaling, using vectorized operations."""
    alpha_H, alpha_Pi, alpha_log = params["alpha_H"], params["alpha_Pi"], params["alpha_log"]
    entropy = alpha_H * (-np.sum(alpha_Pi * probabilities * np.log(alpha_log * probabilities)))
    logging.debug(f"Shannon Entropy: {entropy}")
    print(f"Shannon Entropy: {entropy}")
    return entropy

def mutual_information(H_X: float, H_Y: float, H_XY: float, params: Dict[str, float]) -> float:
    """Calculate Mutual Information with parameter scaling."""
    alpha_I, alpha_HX, alpha_HY, alpha_HXY = params["alpha_I"], params["alpha_HX"], params["alpha_HY"], params["alpha_HXY"]
    mutual_info = alpha_I * (alpha_HX * H_X + alpha_HY * H_Y - alpha_HXY * H_XY)
    logging.debug(f"Mutual Information: {mutual_info}")
    print(f"Mutual Information: {mutual_info}")
    return mutual_info

def operational_efficiency(P: float, E: float, params: Dict[str, float]) -> float:
    """Calculate Operational Efficiency with parameter scaling."""
    alpha_O, alpha_P, alpha_E = params["alpha_O"], params["alpha_P"], params["alpha_E"]
    efficiency = alpha_O * (alpha_P * P / (alpha_E * E))
    logging.debug(f"Operational Efficiency: {efficiency}")
    print(f"Operational Efficiency: {efficiency}")
    return efficiency

def error_management(error_detection_rate: float, correction_capability: float, params: Dict[str, float]) -> float:
    """Calculate Error Management Capability with parameter scaling."""
    alpha_Em, alpha_Error_Detection, alpha_Correction = params["alpha_Em"], params["alpha_Error_Detection"], params["alpha_Correction"]
    error_management_value = alpha_Em * (alpha_Error_Detection * error_detection_rate * alpha_Correction * correction_capability)
    logging.debug(f"Error Management Capability: {error_management_value}")
    print(f"Error Management Capability: {error_management_value}")
    return error_management_value

def adaptability(adaptation_rate: float, params: Dict[str, float]) -> float:
    """Calculate Adaptability with parameter scaling."""
    alpha_A, alpha_Adaptation_Rate = params["alpha_A"], params["alpha_Adaptation_Rate"]
    adaptability_value = alpha_A * (alpha_Adaptation_Rate * adaptation_rate)
    logging.debug(f"Adaptability: {adaptability_value}")
    print(f"Adaptability: {adaptability_value}")
    return adaptability_value

def volume(spatial_scale: float, params: Dict[str, float]) -> float:
    """Calculate Volume with parameter scaling."""
    alpha_Volume, alpha_Spatial_Scale = params["alpha_Volume"], params["alpha_Spatial_Scale"]
    volume_value = alpha_Volume * (alpha_Spatial_Scale * spatial_scale)
    logging.debug(f"Volume: {volume_value}")
    print(f"Volume: {volume_value}")
    return volume_value

def time(temporal_scale: float, params: Dict[str, float]) -> float:
    """Calculate Time with parameter scaling."""
    alpha_t, alpha_Temporal_Scale = params["alpha_t"], params["alpha_Temporal_Scale"]
    time_value = alpha_t * (alpha_Temporal_Scale * temporal_scale)
    logging.debug(f"Time: {time_value}")
    print(f"Time: {time_value}")
    return time_value

def validate_params(params: Dict[str, float]) -> bool:
    """Validate input parameters to ensure they are within expected ranges and types."""
    for key, value in params.items():
        if not isinstance(value, (int, float)):
            logging.error(f"Invalid parameter type for {key}: {type(value)}. Expected int or float.")
            print(f"Invalid parameter type for {key}: {type(value)}. Expected int or float.")
            return False
        if key.startswith("alpha_"):
            if not 0.0 <= value <= 1000.0:       
                logging.error(f"Parameter {key} is out of range: {value}. Expected value between 0.0 and 10.0.")
                print(f"Parameter {key} is out of range: {value}. Expected value between 0.0 and 10.0.")
                return False
        else:
            if not 0.0 <= value <= 1000000.0:
                logging.error(f"Parameter {key} is out of range: {value}. Expected value between 0.0 and 1000000.0.")
                print(f"Parameter {key} is out of range: {value}. Expected value between 0.0 and 1000000.0.")
                return False
    return True

def calculate_intelligence_for_region(region_params: Dict[str, float], probabilities: np.ndarray, H_X: float, H_Y: float, H_XY: float, P: float, E: float, error_detection_rate: float, correction_capability: float, adaptation_rate: float, spatial_scale: float, temporal_scale: float) -> Tuple[float, float, float, float, float, float, float, float]:
    """Calculate each component and overall intelligence for a specific region."""
    logging.debug(f"Calculating for region: {region_params}")
    print(f"Calculating for region: {region_params}")
    try:
        # Validate parameters
        if not validate_params(region_params):
            raise ValueError("Invalid parameters")

        # Calculate each component
        H_X_value = shannon_entropy(probabilities, region_params)
        I_XY_value = mutual_information(H_X, H_Y, H_XY, region_params)
        O_value = operational_efficiency(P, E, region_params)
        Em_value = error_management(error_detection_rate, correction_capability, region_params)
        A_value = adaptability(adaptation_rate, region_params)
        Volume_value = volume(spatial_scale, region_params)
        Time_value = time(temporal_scale, region_params)

        # Calculate intelligence
        I = region_params["k"] * (H_X_value * I_XY_value * O_value * Em_value * A_value) / (Volume_value * Time_value)
        logging.info(f"    Calculated Intelligence: {I}")
        print(f"    Calculated Intelligence: {I}")
        return H_X_value, I_XY_value, O_value, Em_value, A_value, Volume_value, Time_value, I
    except ValueError as ve:
        logging.error(f"    Error calculating intelligence for region: {ve}")
        print(f"    Error calculating intelligence for region: {ve}")
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    except Exception as e:
        logging.error(f"    Unexpected error calculating intelligence for region: {e}")
        print(f"    Unexpected error calculating intelligence for region: {e}")
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

def calculate_intelligence(params_ranges: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    """
    Calculate each component and overall intelligence for all parameters defined.
    This function handles parameter validation and error recovery.
    """
    # Example values for each component
    probabilities = np.array([0.2, 0.3, 0.5])  # Example probabilities for entropy calculation
    H_X = 1.0  # Example entropy of X
    H_Y = 1.0  # Example entropy of Y
    H_XY = 1.5  # Example joint entropy of X and Y
    P = 1000.0  # Example performance value
    E = 50.0  # Example energy value
    error_detection_rate = 0.9  # Example error detection rate
    correction_capability = 0.8  # Example correction capability
    adaptation_rate = 0.7  # Example adaptation rate
    spatial_scale = 1.0  # Example spatial scale
    temporal_scale = 1.0  # Example temporal scale

    for system, regions in params_ranges.items():
        logging.info(f"Calculating for system: {system}")
        print(f"Calculating for system: {system}")
        for region, region_params in regions.items():
            logging.info(f"  Region: {region}")
            print(f"  Region: {region}")
            # Merge default params with region-specific params
            merged_params = {**default_params, **region_params}

            # Ensure all parameters are single numeric values
            for key, value in merged_params.items():
                if isinstance(value, list):
                    merged_params[key] = np.mean(value)  # Use the mean of the list as a single value

            # Calculate intelligence for the region
            H_X_value, I_XY_value, O_value, Em_value, A_value, Volume_value, Time_value, I = calculate_intelligence_for_region(merged_params, probabilities, H_X, H_Y, H_XY, P, E, error_detection_rate, correction_capability, adaptation_rate, spatial_scale, temporal_scale)

# Example parameter ranges for different systems and regions with detailed comments
params_ranges: Dict[str, Dict[str, Dict[str, float]]] = {
    "human": {
        "general": {"alpha_H": [3.0, 4.0], "alpha_I": [0.5, 1.0], "alpha_O": [500, 1000], "alpha_Em": [0.8, 0.9], "alpha_A": [0.7, 0.8], "alpha_Volume": [1.0, 1.5], "alpha_t": [1.0, 60.0]},
"prefrontal_cortex": {"alpha_H": [3.5, 4.0], "alpha_I": [0.8, 1.0], "alpha_O": [500, 1000], "alpha_Em": [0.9, 0.95], "alpha_A": [0.85, 0.9], "alpha_Volume": [1.0, 1.5], "alpha_t": [1.0, 60.0]},
        "motor_cortex": {"alpha_H": [3.0, 3.5], "alpha_I": [0.6, 0.8], "alpha_O": [400, 600], "alpha_Em": [0.8, 0.85], "alpha_A": [0.75, 0.85], "alpha_Volume": [1.0, 1.5], "alpha_t": [1.0, 60.0]},
        "visual_cortex": {"alpha_H": [3.5, 4.5], "alpha_I": [0.9, 1.0], "alpha_O": [800, 1200], "alpha_Em": [0.95, 0.98], "alpha_A": [0.9, 0.95], "alpha_Volume": [1.0, 1.5], "alpha_t": [1.0, 60.0]},
        "cerebellum": {"alpha_H": [3.0, 4.0], "alpha_I": [0.7, 0.9], "alpha_O": [600, 800], "alpha_Em": [0.85, 0.9], "alpha_A": [0.8, 0.9], "alpha_Volume": [1.0, 1.5], "alpha_t": [1.0, 60.0]},
        "hippocampus": {"alpha_H": [3.5, 4.5], "alpha_I": [0.8, 1.0], "alpha_O": [700, 900], "alpha_Em": [0.9, 0.95], "alpha_A": [0.85, 0.95], "alpha_Volume": [1.0, 1.5], "alpha_t": [1.0, 60.0]},
        "thalamus": {"alpha_H": [3.0, 4.0], "alpha_I": [0.6, 0.8], "alpha_O": [500, 700], "alpha_Em": [0.8, 0.9], "alpha_A": [0.75, 0.85], "alpha_Volume": [1.0, 1.5], "alpha_t": [1.0, 60.0]},
        "brainstem": {"alpha_H": [2.5, 3.5], "alpha_I": [0.4, 0.6], "alpha_O": [300, 500], "alpha_Em": [0.7, 0.8], "alpha_A": [0.6, 0.7], "alpha_Volume": [1.0, 1.5], "alpha_t": [1.0, 60.0]},
        "spinal_cord": {"alpha_H": [2.0, 3.0], "alpha_I": [0.2, 0.4], "alpha_O": [200, 400], "alpha_Em": [0.6, 0.7], "alpha_A": [0.5, 0.6], "alpha_Volume": [1.0, 1.5], "alpha_t": [1.0, 60.0]}
    },
    "octopus": {
        "general": {"alpha_H": [3.0, 4.5], "alpha_I": [0.1, 1.0], "alpha_O": [100, 1000], "alpha_Em": [0.8, 0.95], "alpha_A": [0.8, 1.0], "alpha_Volume": [0.1, 0.5], "alpha_t": [1.0, 30.0]},
        "central_brain": {"alpha_H": [3.5, 4.5], "alpha_I": [0.8, 1.0], "alpha_O": [800, 1000], "alpha_Em": [0.85, 0.95], "alpha_A": [0.9, 1.0], "alpha_Volume": [0.1, 0.5], "alpha_t": [1.0, 30.0]},
        "peripheral_nervous_system": {"alpha_H": [3.0, 3.5], "alpha_I": [0.4, 0.6], "alpha_O": [200, 400], "alpha_Em": [0.8, 0.85], "alpha_A": [0.8, 0.9], "alpha_Volume": [0.1, 0.5], "alpha_t": [1.0, 30.0]}
    }
}

# Example: Accessing parameters for human prefrontal cortex
human_prefrontal_cortex_params = params_ranges["human"]["prefrontal_cortex"]
logging.info(f"Human Prefrontal Cortex Parameters: {human_prefrontal_cortex_params}")
print(f"Human Prefrontal Cortex Parameters: {human_prefrontal_cortex_params}")

# Example: Accessing general parameters for octopus
octopus_general_params = params_ranges["octopus"]["general"]
logging.info(f"Octopus General Parameters: {octopus_general_params}")
print(f"Octopus General Parameters: {octopus_general_params}")

# Run the calculation for all defined parameters
calculate_intelligence(params_ranges)