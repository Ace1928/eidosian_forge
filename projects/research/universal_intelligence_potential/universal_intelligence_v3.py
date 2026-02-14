import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Dict, Any, Union, Optional
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joypy  # Library for ridge plots
from tqdm import tqdm
import pyopencl as cl
import pyopencl.array as cl_array
import sys
import cProfile
import pstats

# Set environment variable to see compiler output
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

# Configure logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)

# Define the parameters for the model with explicit type annotations
alpha_parameters: Dict[str, float] = {
    "k": 1.0, "alpha_H": 1.0, "alpha_I": 1.0, "alpha_O": 1.0, "alpha_Em": 1.0,
    "alpha_A": 1.0, "alpha_Volume": 1.0, "alpha_t": 1.0, "alpha_Pi": 1.0,
    "alpha_log": 1.0, "alpha_HX": 1.0, "alpha_HY": 1.0, "alpha_HXY": 1.0,
    "alpha_P": 1.0, "alpha_E": 1.0, "alpha_Error_Detection": 1.0,
    "alpha_Correction": 1.0, "alpha_Adaptation_Rate": 1.0, "alpha_Spatial_Scale": 1.0,
    "alpha_Temporal_Scale": 1.0
}

def log_info(message: str) -> None:
    """Log an info message."""
    logging.info(f"INFO: {message}")

def log_debug(message: str) -> None:
    """Log a debug message."""
    logging.debug(f"DEBUG: {message}")

def log_error(message: str) -> None:
    """Log an error message."""
    logging.error(f"ERROR: {message}")

log_info(f"Alpha parameters: {alpha_parameters}")

def setup_opencl_environment() -> Tuple[cl.Context, cl.CommandQueue]:
    """
    Setup the OpenCL environment and return the context and command queue.
    
    Returns:
        Tuple[cl.Context, cl.CommandQueue]: A tuple containing the OpenCL context and command queue.
    
    Raises:
        Exception: If an error occurs during the setup.
    """
    try:
        log_debug("Attempting to get OpenCL platforms.")
        platform = cl.get_platforms()[0]
        log_debug(f"Platform obtained: {platform}")
        
        log_debug("Attempting to get OpenCL devices from platform.")
        device = platform.get_devices()[0]
        log_debug(f"Device obtained: {device}")
        
        log_debug("Creating OpenCL context with the obtained device.")
        context = cl.Context([device])
        log_debug("OpenCL context created successfully.")
        
        log_debug("Creating OpenCL command queue with the created context.")
        queue = cl.CommandQueue(context)
        log_debug("OpenCL command queue created successfully.")
        
        return context, queue
    except Exception as e:
        log_error(f"Error setting up OpenCL environment: {e}")
        raise

def load_kernel(filename: str) -> str:
    """
    Load OpenCL kernel code from a file.
    
    Parameters:
        filename (str): The filename of the kernel code file.
    
    Returns:
        str: The kernel code as a string.
    
    Raises:
        IOError: If the file cannot be opened.
        Exception: For any other issues that arise.
    """
    try:
        log_debug(f"Attempting to open kernel file: {filename}")
        with open(filename, 'r') as file:
            kernel_code = file.read()
        log_debug(f"Kernel loaded successfully from {filename}")
        return kernel_code
    except IOError as e:
        log_error(f"Error opening kernel file {filename}: {e}")
        raise
    except Exception as e:
        log_error(f"Unexpected error loading kernel from {filename}: {e}")
        raise

def initialize_progress_bar(total_operations: int) -> tqdm:
    """
    Initialize the progress bar.
    
    Parameters:
        total_operations (int): The total number of operations.
    
    Returns:
        tqdm: The initialized progress bar.
    """
    log_debug(f"Initializing progress bar with total operations: {total_operations}")
    return tqdm(total=total_operations, desc="Processing", unit="operation")

def load_and_build_program(context: cl.Context, kernel_path: str) -> cl.Program:
    """
    Load OpenCL kernel code from a file and build the program.
    
    Parameters:
        context (cl.Context): The OpenCL context.
        kernel_path (str): The path to the kernel file.
    
    Returns:
        cl.Program: The built OpenCL program.
    
    Raises:
        Exception: If there is an error in loading the kernel or building the program.
    """
    try:
        log_debug(f"Loading kernel from path: {kernel_path}")
        kernel_code = load_kernel(kernel_path)
        
        log_debug(f"Building OpenCL program from kernel code.")
        program = cl.Program(context, kernel_code).build()
        log_debug(f"Program built successfully from {kernel_path}")
        
        return program
    except Exception as e:
        log_error(f"Failed to load and build program from {kernel_path}: {e}")
        raise

try:
    log_debug("Setting up OpenCL environment.")
    CONTEXT, QUEUE = setup_opencl_environment()
    log_debug("OpenCL environment setup completed.")
except Exception as e:
    log_error(f"Failed to set up OpenCL environment: {e}")
    raise

GROUP_SIZE: int = 256
kernel_directory: str = "/home/lloyd/UniversalIntelligencePotential/kernels/"

try:
    log_debug("Loading and building OpenCL programs.")
    entropy_program = load_and_build_program(CONTEXT, kernel_directory + 'entropy_kernel.cl')
    mutual_information_program = load_and_build_program(CONTEXT, kernel_directory + 'mutual_information_kernel.cl')
    operational_efficiency_program = load_and_build_program(CONTEXT, kernel_directory + 'operational_efficiency_kernel.cl')
    error_management_program = load_and_build_program(CONTEXT, kernel_directory + 'error_management_kernel.cl')
    adaptability_program = load_and_build_program(CONTEXT, kernel_directory + 'adaptability_kernel.cl')
    volume_program = load_and_build_program(CONTEXT, kernel_directory + 'volume_kernel.cl')
    time_program = load_and_build_program(CONTEXT, kernel_directory + 'time_kernel.cl')
    log_debug("All OpenCL programs loaded and built successfully.")
except Exception as e:
    log_error(f"Error in loading and building programs: {e}")
    raise

def calculate_total_operations(parameter_ranges: Dict[str, np.ndarray]) -> int:
    """
    Calculate the total number of operations based on the number of parameters and their values,
    and then use those values to calculate all possible combinations and process each combination
    through each metric/calculation function to get the total number of operations.
    
    Parameters:
        parameter_ranges (Dict[str, np.ndarray]): The loaded parameter ranges.
        
    Returns:
        int: The total number of operations.
    """
    log_debug("Calculating total number of operations based on parameter ranges.")
    
    # Validate that all parameter sets are numpy arrays
    for key, value in parameter_ranges.items():
        if not isinstance(value, np.ndarray):
            log_error(f"Expected parameter set {key} to be a numpy array.")
            raise TypeError(f"Expected parameter set {key} to be a numpy array.")
    
    # Generate all possible combinations of parameter values
    keys = list(parameter_ranges.keys())
    values = list(parameter_ranges.values())
    combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
    
    log_debug(f"Generated {len(combinations)} combinations of parameter values.")
    
    # Process each combination through each metric/calculation function
    total_operations = 0
    for combination in combinations:
        try:
            # Extract individual parameters
            probabilities = combination['probabilities_set']
            H_X = combination['H_X_set']
            H_Y = combination['H_Y_set']
            H_XY = combination['H_XY_set']
            P = combination['P_set']
            E = combination['E_set']
            error_detection_rate = combination['error_detection_rate_set']
            correction_capability = combination['correction_capability_set']
            adaptation_rate = combination['adaptation_rate_set']
            spatial_scale = combination['spatial_scale_set']
            temporal_scale = combination['temporal_scale_set']

            # Calculate metrics
            H_X_value = calculate_entropy(probabilities, alpha_parameters, CONTEXT, QUEUE)
            I_XY_value = calculate_mutual_information(H_X, H_Y, H_XY, alpha_parameters, CONTEXT, QUEUE)
            O_value = calculate_operational_efficiency(P, E, alpha_parameters, CONTEXT, QUEUE)
            Em_value = calculate_error_management(error_detection_rate, correction_capability, alpha_parameters, CONTEXT, QUEUE)
            A_value = calculate_adaptability(adaptation_rate, alpha_parameters, CONTEXT, QUEUE)
            Volume_value = calculate_volume(spatial_scale, alpha_parameters, CONTEXT, QUEUE)
            Time_value = calculate_time(temporal_scale, alpha_parameters, CONTEXT, QUEUE)

            # Compute intelligence metric
            intelligence_metric = alpha_parameters["k"] * (H_X_value * I_XY_value * O_value * Em_value * A_value) / (Volume_value * Time_value)
            
            if intelligence_metric is not None:
                total_operations += 1
        except ValueError as ve:
            log_error(f"Validation error for combination {combination}: {ve}")
        except Exception as e:
            log_error(f"Failed to compute intelligence for combination {combination}: {e}")
    
    log_debug(f"Total number of operations calculated: {total_operations}")
    return total_operations

def generate_combinations(parameter_ranges: Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]) -> List[Dict[str, Any]]:
    """
    Generate all possible combinations of parameter values.
    
    Parameters:
        parameter_ranges (Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]): A dictionary containing parameter ranges as numpy arrays or dictionaries of numpy arrays.
        
    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each representing a unique combination of parameter values.
    """
    log_debug("Generating all possible combinations of parameter values.")
    keys = list(parameter_ranges.keys())
    values = []
    for value in parameter_ranges.values():
        if isinstance(value, np.ndarray):
            values.append(value)
        elif isinstance(value, dict):
            values.append(list(value.values()))
        else:
            log_error(f"Unexpected type for parameter value: {type(value)}")
            raise TypeError(f"Unexpected type for parameter value: {type(value)}")
    combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
    log_debug(f"Generated {len(combinations)} combinations of parameter values.")
    return combinations

def calculate_entropy(probabilities: Union[np.ndarray, List[float]], alpha_params: Dict[str, float], context: cl.Context, queue: cl.CommandQueue) -> float:
    """
    Calculate the Shannon entropy of a probability distribution using scaling factors and OpenCL for parallel processing.
    
    Parameters:
        probabilities (Union[np.ndarray, List[float]]): The probability distribution array or list.
        alpha_params (Dict[str, float]): Dictionary of alpha parameters for scaling.
        context (cl.Context): The OpenCL context.
        queue (cl.CommandQueue): The OpenCL command queue.
    
    Returns:
        float: The calculated Shannon entropy.
    
    Raises:
        Exception: If an error occurs during the calculation.
    """
    try:
        probabilities = np.array(probabilities, dtype=np.float32)
        alpha_values = np.array([alpha_params["alpha_H"], alpha_params["alpha_Pi"], alpha_params["alpha_log"]], dtype=np.float32)
        prob_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=probabilities)
        alpha_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=alpha_values)
        result_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, probabilities.nbytes)
        entropy_program.calculate_entropy(queue, probabilities.shape, None, prob_buf, alpha_buf, result_buf, np.int32(len(probabilities)))
        result = np.empty_like(probabilities)
        cl.enqueue_copy(queue, result, result_buf).wait()
        entropy = np.sum(result)
        logging.debug(f"calculate_entropy: Calculated Shannon entropy: {entropy}")
        return entropy
    except Exception as e:
        logging.error(f"calculate_entropy: Error calculating Shannon entropy: {e}")
        raise

def calculate_mutual_information(H_X: Union[float, np.ndarray], H_Y: Union[float, np.ndarray], H_XY: Union[float, np.ndarray], alpha_params: Dict[str, float], context: cl.Context, queue: cl.CommandQueue) -> float:
    """
    Calculate the mutual information based on entropies of X, Y, and their joint distribution using OpenCL for parallel processing.
    
    Parameters:
        H_X (Union[float, np.ndarray]): Entropy of X.
        H_Y (Union[float, np.ndarray]): Entropy of Y.
        H_XY (Union[float, np.ndarray]): Joint entropy of X and Y.
        alpha_params (Dict[str, float]): Dictionary of alpha parameters for scaling.
        context (cl.Context): The OpenCL context.
        queue (cl.CommandQueue): The OpenCL command queue.
    
    Returns:
        float: The calculated mutual information.
    
    Raises:
        Exception: If an error occurs during the calculation.
    """
    try:
        H_X = np.array(H_X, dtype=np.float32)
        H_Y = np.array(H_Y, dtype=np.float32)
        H_XY = np.array(H_XY, dtype=np.float32)
        alpha_values = np.array([alpha_params["alpha_I"], alpha_params["alpha_HX"], alpha_params["alpha_HY"], alpha_params["alpha_HXY"]], dtype=np.float32)
        H_X_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=H_X)
        H_Y_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=H_Y)
        H_XY_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=H_XY)
        alpha_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=alpha_values)
        result_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, H_X.nbytes)
        mutual_information_program.calculate_mutual_information(queue, H_X.shape, None, H_X_buf, H_Y_buf, H_XY_buf, alpha_buf, result_buf)
        result = np.empty_like(H_X)
        cl.enqueue_copy(queue, result, result_buf).wait()
        mutual_info = np.sum(result)
        logging.debug(f"calculate_mutual_information: Calculated mutual information: {mutual_info}")
        return mutual_info
    except Exception as e:
        logging.error(f"calculate_mutual_information: Error calculating mutual information: {e}")
        raise

def calculate_operational_efficiency(P: Union[float, np.ndarray], E: Union[float, np.ndarray], alpha_params: Dict[str, float], context: cl.Context, queue: cl.CommandQueue) -> float:
    """
    Calculate the operational efficiency based on performance and energy consumption using OpenCL for parallel processing.
    
    Parameters:
        P (Union[float, np.ndarray]): Performance measure, typically computational power or output rate.
        E (Union[float, np.ndarray]): Energy consumption measure.
        alpha_params (Dict[str, float]): Dictionary of alpha parameters for scaling.
        context (cl.Context): The OpenCL context.
        queue (cl.CommandQueue): The OpenCL command queue.
        
    Returns:
        float: The calculated operational efficiency.
        
    Raises:
        Exception: If an error occurs during the calculation.
    """
    try:
        P = np.asarray(P, dtype=np.float32)
        E = np.asarray(E, dtype=np.float32)
        alpha_values = np.array([alpha_params["alpha_O"], alpha_params["alpha_P"], alpha_params["alpha_E"]], dtype=np.float32)
        
        P_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=P)
        E_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=E)
        alpha_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=alpha_values)
        result_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, P.nbytes)
        
        operational_efficiency_program.calculate_operational_efficiency(queue, P.shape, None, P_buf, E_buf, alpha_buf, result_buf)
        
        result = np.empty_like(P)
        cl.enqueue_copy(queue, result, result_buf).wait()
        efficiency = np.sum(result)
        
        logging.debug(f"calculate_operational_efficiency: Calculated operational efficiency: {efficiency}")
        return efficiency
    except Exception as e:
        logging.error(f"calculate_operational_efficiency: Error calculating operational efficiency: {e}")
        raise

def calculate_error_management(error_detection_rate: Union[float, np.ndarray], correction_capability: Union[float, np.ndarray], alpha_params: Dict[str, float], context: cl.Context, queue: cl.CommandQueue) -> float:
    """
    Calculate the error management effectiveness based on error detection and correction capabilities using OpenCL for parallel processing.
    
    Parameters:
        error_detection_rate (Union[float, np.ndarray]): The rate at which errors are detected.
        correction_capability (Union[float, np.ndarray]): The capability of the system to correct detected errors.
        alpha_params (Dict[str, float]): Dictionary of alpha parameters for scaling.
        context (cl.Context): The OpenCL context.
        queue (cl.CommandQueue): The OpenCL command queue.
        
    Returns:
        float: The calculated error management effectiveness.
        
    Raises:
        Exception: If an error occurs during the calculation.
    """
    try:
        error_detection_rate = np.asarray(error_detection_rate, dtype=np.float32)
        correction_capability = np.asarray(correction_capability, dtype=np.float32)
        alpha_values = np.array([alpha_params["alpha_Em"], alpha_params["alpha_Error_Detection"], alpha_params["alpha_Correction"]], dtype=np.float32)
        
        error_detection_rate_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=error_detection_rate)
        correction_capability_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=correction_capability)
        alpha_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=alpha_values)
        result_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, error_detection_rate.nbytes)
        
        error_management_program.calculate_error_management(queue, error_detection_rate.shape, None, error_detection_rate_buf, correction_capability_buf, alpha_buf, result_buf)
        
        result = np.empty_like(error_detection_rate)
        cl.enqueue_copy(queue, result, result_buf).wait()
        error_management_value = float(np.sum(result))
        
        logging.debug(f"calculate_error_management: Calculated error management effectiveness: {error_management_value}")
        return error_management_value
    except Exception as e:
        logging.error(f"calculate_error_management: Error calculating error management effectiveness: {e}")
        raise

def calculate_adaptability(adaptation_rate: Union[float, np.ndarray], alpha_params: Dict[str, float], context: cl.Context, queue: cl.CommandQueue) -> float:
    """
    Calculate the adaptability based on the adaptation rate using OpenCL for parallel processing.
    
    Parameters:
        adaptation_rate (Union[float, np.ndarray]): The rate at which the system can adapt to changes.
        alpha_params (Dict[str, float]): Dictionary of alpha parameters for scaling.
        context (cl.Context): The OpenCL context.
        queue (cl.CommandQueue): The OpenCL command queue.
        
    Returns:
        float: The calculated adaptability.
        
    Raises:
        Exception: If an error occurs during the calculation.
    """
    try:
        adaptation_rate = np.asarray(adaptation_rate, dtype=np.float32)
        alpha_values = np.array([alpha_params["alpha_A"], alpha_params["alpha_Adaptation_Rate"]], dtype=np.float32)
        
        adaptation_rate_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=adaptation_rate)
        alpha_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=alpha_values)
        result_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, adaptation_rate.nbytes)
        
        adaptability_program.calculate_adaptability(queue, adaptation_rate.shape, None, adaptation_rate_buf, alpha_buf, result_buf)
        
        result = np.empty_like(adaptation_rate)
        cl.enqueue_copy(queue, result, result_buf).wait()
        adaptability_value = float(np.sum(result))
        
        logging.debug(f"calculate_adaptability: Calculated adaptability: {adaptability_value}")
        return adaptability_value
    except Exception as e:
        logging.error(f"calculate_adaptability: Error calculating adaptability: {e}")
        raise

def calculate_volume(spatial_scale: Union[float, np.ndarray], alpha_params: Dict[str, float], context: cl.Context, queue: cl.CommandQueue) -> float:
    """
    Calculate the volume based on the spatial scale using OpenCL for parallel processing.
    
    Parameters:
        spatial_scale (Union[float, np.ndarray]): The spatial scale factor, typically representing physical dimensions.
        alpha_params (Dict[str, float]): Dictionary of alpha parameters for scaling.
        context (cl.Context): The OpenCL context.
        queue (cl.CommandQueue): The OpenCL command queue.
        
    Returns:
        float: The calculated volume.
        
    Raises:
        Exception: If an error occurs during the calculation.
    """
    try:
        spatial_scale = np.asarray(spatial_scale, dtype=np.float32)
        alpha_values = np.array([alpha_params["alpha_Volume"], alpha_params["alpha_Spatial_Scale"]], dtype=np.float32)
        
        spatial_scale_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=spatial_scale)
        alpha_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=alpha_values)
        result_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, spatial_scale.nbytes)
        
        volume_program.calculate_volume(queue, spatial_scale.shape, None, spatial_scale_buf, alpha_buf, result_buf)
        
        result = np.empty_like(spatial_scale)
        cl.enqueue_copy(queue, result, result_buf).wait()
        volume_value = float(np.sum(result))
        
        logging.debug(f"calculate_volume: Calculated volume: {volume_value}")
        return volume_value
    except Exception as e:
        logging.error(f"calculate_volume: Error calculating volume: {e}")
        raise

def calculate_time(temporal_scale: Union[float, np.ndarray], alpha_params: Dict[str, float], context: cl.Context, queue: cl.CommandQueue) -> float:
    """
    Calculate the time based on the temporal scale using OpenCL for parallel processing.
    
    Parameters:
        temporal_scale (Union[float, np.ndarray]): The temporal scale factor, typically representing time dimensions.
        alpha_params (Dict[str, float]): Dictionary of alpha parameters for scaling.
        context (cl.Context): The OpenCL context.
        queue (cl.CommandQueue): The OpenCL command queue.
        
    Returns:
        float: The calculated time.
        
    Raises:
        Exception: If an error occurs during the calculation.
    """
    try:
        temporal_scale = np.asarray(temporal_scale, dtype=np.float32)
        alpha_values = np.array([alpha_params["alpha_t"], alpha_params["alpha_Temporal_Scale"]], dtype=np.float32)
        
        temporal_scale_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=temporal_scale)
        alpha_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=alpha_values)
        result_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, temporal_scale.nbytes)
        
        time_program.calculate_time(queue, temporal_scale.shape, None, temporal_scale_buf, alpha_buf, result_buf)
        
        result = np.empty_like(temporal_scale)
        cl.enqueue_copy(queue, result, result_buf).wait()
        time_value = float(np.sum(result))
        
        logging.debug(f"calculate_time: Calculated time: {time_value}")
        return time_value
    except Exception as e:
        logging.error(f"calculate_time: Error calculating time: {e}")
        raise

def compute_intelligence_v2(combination: Dict[str, float], context: cl.Context, queue: cl.CommandQueue) -> Optional[float]:
    """
    Compute the intelligence metric for a given parameter combination, handling exceptions and ensuring robust computation.
    
    Parameters:
        combination (Dict[str, float]): The parameter combination.
        context (cl.Context): The OpenCL context.
        queue (cl.CommandQueue): The OpenCL command queue.
    
    Returns:
        Optional[float]: The computed intelligence metric or None if an error occurs.
    """
    log_debug(f"Starting computation of intelligence metric with combination: {combination}")

    def extract_parameters(combination: Dict[str, float]) -> Dict[str, float]:
        """
        Extract parameters from the combination dictionary with default values.
        
        Parameters:
            combination (Dict[str, float]): The parameter combination.
        
        Returns:
            Dict[str, float]: The extracted parameters with default values.
        """
        extracted_params = {
            "H_X_value": combination.get("H_X_value", 1.0),
            "I_XY_value": combination.get("I_XY_value", 1.0),
            "O_value": combination.get("O_value", 1.0),
            "Em_value": combination.get("Em_value", 1.0),
            "A_value": combination.get("A_value", 1.0),
            "Volume_value": combination.get("Volume_value", 1.0),
            "Time_value": combination.get("Time_value", 1.0),
            "k": combination.get("k", 1.0)
        }
        log_debug(f"Extracted parameters: {extracted_params}")
        return extracted_params

    def validate_parameters(parameters: Dict[str, float]) -> None:
        """
        Validate the parameters to ensure non-zero and non-negligible denominators.
        
        Parameters:
            parameters (Dict[str, float]): The parameters to validate.
        
        Raises:
            ValueError: If Volume_value or Time_value are zero or negligible.
        """
        log_debug(f"Validating parameters: {parameters}")
        if np.isclose(parameters["Volume_value"], 0) or np.isclose(parameters["Time_value"], 0):
            log_error("Volume_value and Time_value must be non-zero and non-negligible.")
            raise ValueError("Volume_value and Time_value must be non-zero and non-negligible.")
        log_debug("Parameter validation passed.")

    def compute_metric(parameters: Dict[str, float]) -> float:
        """
        Compute the intelligence metric using the provided parameters.
        
        Parameters:
            parameters (Dict[str, float]): The parameters for computation.
        
        Returns:
            float: The computed intelligence metric.
        """
        log_debug(f"Computing intelligence metric with parameters: {parameters}")
        metric = parameters["k"] * (
            parameters["H_X_value"] * parameters["I_XY_value"] * parameters["O_value"] * 
            parameters["Em_value"] * parameters["A_value"]
        ) / (parameters["Volume_value"] * parameters["Time_value"])
        log_debug(f"Computed intelligence metric: {metric}")
        return metric

    try:
        parameters: Dict[str, float] = extract_parameters(combination)
        validate_parameters(parameters)
        intelligence_metric: float = compute_metric(parameters)
        log_debug(f"Computed intelligence metric: {intelligence_metric}")
        return intelligence_metric

    except ValueError as ve:
        log_error(f"Validation error for combination {combination}: {ve}")
        return None
    except Exception as e:
        log_error(f"Failed to compute intelligence for combination {combination}: {e}")
        return None


def load_parameter_ranges() -> Dict[str, np.ndarray]:
    """
    Load or define the parameter ranges for the model.
    
    Returns:
        Dict[str, np.ndarray]: A dictionary containing the parameter ranges as numpy arrays.
    """
    if os.path.exists('parameter_ranges.npz'):
        with np.load('parameter_ranges.npz', allow_pickle=True) as data:
            return {
                'probabilities_set': data['probabilities_set'],
                'H_X_set': data['H_X_set'],
                'H_Y_set': data['H_Y_set'],
                'H_XY_set': data['H_XY_set'],
                'P_set': data['P_set'],
                'E_set': data['E_set'],
                'error_detection_rate_set': data['error_detection_rate_set'],
                'correction_capability_set': data['correction_capability_set'],
                'adaptation_rate_set': data['adaptation_rate_set'],
                'spatial_scale_set': data['spatial_scale_set'],
                'temporal_scale_set': data['temporal_scale_set']
            }
    else:
        probabilities_set = {
            f'prob_{p1}_{p2}': np.array([p1, p2, 1 - p1 - p2])
            for p1 in np.linspace(0.1, 0.9, 9)
            for p2 in np.linspace(0.1, 0.9, 9)
            if p1 + p2 <= 1
        }
        H_X_set = np.linspace(0.2, 2.0, 5)
        H_Y_set = np.linspace(0.2, 2.0, 5)
        H_XY_set = np.linspace(0.5, 3.0, 5)
        P_set = np.linspace(100.0, 2000.0, 5)
        E_set = np.linspace(20.0, 100.0, 5)
        error_detection_rate_set = np.linspace(0.5, 1.0, 3)
        correction_capability_set = np.linspace(0.5, 1.0, 3)
        adaptation_rate_set = np.linspace(0.3, 1.0, 4)
        spatial_scale_set = np.linspace(0.5, 1.5, 3)
        temporal_scale_set = np.linspace(0.5, 1.5, 3)

        np.savez('parameter_ranges.npz', 
                 probabilities_set=probabilities_set, 
                 H_X_set=H_X_set, 
                 H_Y_set=H_Y_set, 
                 H_XY_set=H_XY_set,
                 P_set=P_set, 
                 E_set=E_set, 
                 error_detection_rate_set=error_detection_rate_set,
                 correction_capability_set=correction_capability_set, 
                 adaptation_rate_set=adaptation_rate_set,
                 spatial_scale_set=spatial_scale_set, 
                 temporal_scale_set=temporal_scale_set)
        return {
            'probabilities_set': probabilities_set,
            'H_X_set': H_X_set,
            'H_Y_set': H_Y_set,
            'H_XY_set': H_XY_set,
            'P_set': P_set,
            'E_set': E_set,
            'error_detection_rate_set': error_detection_rate_set,
            'correction_capability_set': correction_capability_set,
            'adaptation_rate_set': adaptation_rate_set,
            'spatial_scale_set': spatial_scale_set,
            'temporal_scale_set': temporal_scale_set
        }
    
# Plotting Functions
def plot_heatmap(data: pd.DataFrame) -> None:
    """
    Plot the heatmap of the correlation matrix.

    Parameters:
        data (pd.DataFrame): The data for which the correlation matrix heatmap is to be plotted.
    """
    log_debug(f"Starting to plot heatmap for data with shape {data.shape} and columns {data.columns.tolist()}")
    try:
        plt.figure(figsize=(14, 10))
        log_debug("Figure created with size (14, 10)")
        correlation_matrix = data.corr()
        log_debug(f"Correlation matrix calculated: {correlation_matrix}")
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        log_debug("Heatmap created with 'coolwarm' colormap and linewidths=0.5")
        plt.title('Correlation Matrix of Parameters and Intelligence')
        plt.savefig('correlation_matrix.png')
        log_debug("Heatmap saved as 'correlation_matrix.png'")
        plt.show(block=False)
        plt.pause(5)
        plt.close()
        log_debug("Heatmap displayed and closed successfully.")
    except Exception as e:
        log_error(f"Failed to plot heatmap: {e}")

def plot_pairplot(data: pd.DataFrame) -> None:
    """
    Plot the pairplot of the parameters.

    Parameters:
        data (pd.DataFrame): The data for which the pairplot is to be plotted.
    """
    log_debug(f"Starting to plot pairplot for data with shape {data.shape} and columns {data.columns.tolist()}")
    try:
        sns.pairplot(data, hue='Intelligence', palette='viridis')
        log_debug("Pairplot created with 'Intelligence' as hue and 'viridis' palette")
        plt.savefig('pairplot.png')
        log_debug("Pairplot saved as 'pairplot.png'")
        plt.show(block=False)
        plt.pause(5)
        plt.close()
        log_debug("Pairplot displayed and closed successfully.")
    except Exception as e:
        logging.error(f"Failed to plot pairplot: {e}")

def plot_jointplot(data: pd.DataFrame) -> None:
    """
    Plot the jointplot of the parameters.

    Parameters:
        data (pd.DataFrame): The data for which the jointplot is to be plotted.
    """
    log_debug(f"Starting to plot jointplot for data with shape {data.shape} and columns {data.columns.tolist()}")
    try:
        sns.jointplot(data=data, x='H_X', y='Intelligence', kind='hex', cmap='Blues')
        log_debug("Jointplot created with 'H_X' as x, 'Intelligence' as y, 'hex' kind, and 'Blues' colormap")
        plt.savefig('jointplot.png')
        log_debug("Jointplot saved as 'jointplot.png'")
        plt.show(block=False)
        plt.pause(5)
        plt.close()
        log_debug("Jointplot displayed and closed successfully.")
    except Exception as e:
        logging.error(f"Failed to plot jointplot: {e}")

def plot_histograms(data: pd.DataFrame) -> None:
    """
    Plot the histograms of the parameters.

    Parameters:
        data (pd.DataFrame): The data for which the histograms are to be plotted.
    """
    log_debug(f"Starting to plot histograms for data with shape {data.shape} and columns {data.columns.tolist()}")
    try:
        for column in data.columns:
            log_debug(f"Plotting histogram for column: {column}")
            plt.figure(figsize=(10, 6))
            sns.histplot(data[column], bins=20, kde=True, color='magenta')
            log_debug(f"Histogram created for column {column} with 20 bins, KDE=True, and color 'magenta'")
            plt.title(f'Distribution of {column}')
            plt.savefig(f'{column}_distribution.png')
            log_debug(f"Histogram for column {column} saved as '{column}_distribution.png'")
            plt.show(block=False)
            plt.pause(5)
            plt.close()
            log_debug(f"Histogram for column {column} displayed and closed successfully.")
    except Exception as e:
        logging.error(f"Failed to plot histograms: {e}")

def plot_boxplots(data: pd.DataFrame) -> None:
    """
    Plot the boxplots of the parameters.

    Parameters:
        data (pd.DataFrame): The data for which the boxplots are to be plotted.
    """
    log_debug(f"Starting to plot boxplots for data with shape {data.shape} and columns {data.columns.tolist()}")
    try:
        for column in data.columns[:-1]:
            log_debug(f"Plotting boxplot for column: {column}")
            plt.figure(figsize=(12, 8))
            sns.boxplot(x='Intelligence', y=column, data=data)
            log_debug(f"Boxplot created for column {column} with 'Intelligence' as x-axis")
            plt.title(f'Intelligence vs {column}')
            plt.savefig(f'Intelligence_vs_{column}.png')
            log_debug(f"Boxplot for column {column} saved as 'Intelligence_vs_{column}.png'")
            plt.show(block=False)
            plt.pause(5)
            plt.close()
            log_debug(f"Boxplot for column {column} displayed and closed successfully.")
    except Exception as e:
        logging.error(f"Failed to plot boxplots: {e}")

def plot_violinplot(data: pd.DataFrame) -> None:
    """
    Plot the violinplot of the parameters.

    Parameters:
        data (pd.DataFrame): The data for which the violinplot is to be plotted.
    """
    log_debug(f"Starting to plot violinplot for data with shape {data.shape} and columns {data.columns.tolist()}")
    try:
        plt.figure(figsize=(12, 8))
        sns.violinplot(data=data, x='Error Detection Rate', y='Correction Capability', scale='width', inner='quartile')
        log_debug("Violinplot created with 'Error Detection Rate' as x, 'Correction Capability' as y, scale='width', and inner='quartile'")
        plt.title('Error Detection Rate vs Correction Capability')
        plt.savefig('Error_Detection_vs_Correction_Capability_violin.png')
        log_debug("Violinplot saved as 'Error_Detection_vs_Correction_Capability_violin.png'")
        plt.show(block=False)
        plt.pause(5)
        plt.close()
        log_debug("Violinplot displayed and closed successfully.")
    except Exception as e:
        logging.error(f"Failed to plot violinplot: {e}")

def plot_scatter_matrix(data: pd.DataFrame) -> None:
    """
    Plot the scatter matrix of the parameters.

    Parameters:
        data (pd.DataFrame): The data for which the scatter matrix is to be plotted.
    """
    log_debug(f"Starting to plot scatter matrix for data with shape {data.shape} and columns {data.columns.tolist()}")
    try:
        plt.figure(figsize=(20, 15))
        pd.plotting.scatter_matrix(data, alpha=0.8, figsize=(20, 15), diagonal='kde')
        log_debug("Scatter matrix created with alpha=0.8 and diagonal='kde'")
        plt.savefig('scatter_matrix.png')
        log_debug("Scatter matrix saved as 'scatter_matrix.png'")
        plt.show(block=False)
        plt.pause(5)
        plt.close()
        log_debug("Scatter matrix displayed and closed successfully.")
    except Exception as e:
        logging.error(f"Failed to plot scatter matrix: {e}")

def plot_ridge_plot(data: pd.DataFrame) -> None:
    """
    Plot the ridge plot of the parameters.

    Parameters:
        data (pd.DataFrame): The data for which the ridge plot is to be plotted.
    """
    log_debug(f"Starting to plot ridge plot for data with shape {data.shape} and columns {data.columns.tolist()}")
    try:
        fig, axes = joypy.joyplot(data, by='Intelligence', figsize=(12, 8), colormap=plt.cm.viridis, alpha=0.8)
        log_debug("Ridge plot created with 'Intelligence' as by, figsize=(12, 8), colormap=plt.cm.viridis, and alpha=0.8")
        plt.title('Ridge Plot of Parameters Grouped by Intelligence')
        plt.savefig('ridge_plot.png')
        log_debug("Ridge plot saved as 'ridge_plot.png'")
        plt.show(block=False)
        plt.pause(5)
        plt.close()
        log_debug("Ridge plot displayed and closed successfully.")
    except Exception as e:
        logging.error(f"Failed to plot ridge plot: {e}")

def plot_3d_scatter(data: pd.DataFrame) -> None:
    """
    Plot the 3D scatter plot of the parameters.

    Parameters:
        data (pd.DataFrame): The data for which the 3D scatter plot is to be plotted.
    """
    log_debug(f"Starting to plot 3D scatter plot for data with shape {data.shape} and columns {data.columns.tolist()}")
    try:
        fig = plt.figure(figsize=(14, 10))
        log_debug("3D scatter plot figure created with size (14, 10)")
        ax = fig.add_subplot(111, projection='3d')
        log_debug("3D scatter plot axis created with projection='3d'")
        ax.scatter(data['H_X'], data['P'], data['Intelligence'], c='r', marker='o')
        log_debug("3D scatter plot created with 'H_X' as x, 'P' as y, 'Intelligence' as z, color='r', and marker='o'")
        ax.set_xlabel('H_X')
        ax.set_ylabel('P')
        ax.set_zlabel('Intelligence')
        plt.title('3D Scatter Plot of H_X, P and Intelligence')
        plt.savefig('3D_scatter_H_X_P_Intelligence.png')
        log_debug("3D scatter plot saved as '3D_scatter_H_X_P_Intelligence.png'")
        plt.show(block=False)
        plt.pause(5)
        plt.close()
        log_debug("3D scatter plot displayed and closed successfully.")
    except Exception as e:
        logging.error(f"Failed to plot 3D scatter: {e}")

def visualize_results(results: pd.DataFrame) -> None:
    """
    Visualize the results using various plots.

    Parameters:
        results (pd.DataFrame): The results DataFrame.
    """
    log_debug(f"Starting to visualize results with shape {results.shape} and columns {results.columns.tolist()}")
    try:
        numeric_results = results.drop(columns=['Probabilities'])
        log_debug(f"Dropped 'Probabilities' column, resulting in numeric results with shape {numeric_results.shape} and columns {numeric_results.columns.tolist()}")
        print(results.to_string(index=False))

        plot_heatmap(numeric_results)
        plot_pairplot(numeric_results)
        plot_jointplot(numeric_results)
        plot_histograms(numeric_results)
        plot_boxplots(numeric_results)
        plot_violinplot(numeric_results)
        plot_scatter_matrix(numeric_results)
        plot_ridge_plot(numeric_results)
        plot_3d_scatter(numeric_results)
    except Exception as e:
        logging.error(f"Failed to visualize results: {e}")
    finally:
        logging.info("Visualization process completed.")



def cleanup_resources(context: cl.Context, queue: cl.CommandQueue, profiler: cProfile.Profile, logging_choice: str) -> None:
    """
    Clean up OpenCL resources, profiler, and other system resources to ensure no memory leaks.

    Parameters:
        context (cl.Context): The OpenCL context to be released.
        queue (cl.CommandQueue): The OpenCL command queue to be finished.
        profiler (cProfile.Profile): The profiler instance to be disabled.
        logging_choice (str): The logging choice indicating the level of logging ('V' for Verbose, 'B' for Brief).
    """
    log_debug("Starting resource cleanup process.")
    try:
        # Finish all operations in the command queue
        queue.finish()
        log_debug("OpenCL command queue finished successfully.")
        
        # Correct handling without release
        log_debug("OpenCL context managed by Python's garbage collector.")
        
    except cl.RuntimeError as cl_error:
        logging.error(f"OpenCL runtime error during resource cleanup: {cl_error}")
    except Exception as e:
        logging.error(f"Unexpected error during OpenCL resource cleanup: {e}")
    finally:
        # Handle profiler based on logging choice
        if logging_choice in ['V', 'B']:
            try:
                profiler.disable()
                log_debug("Profiler disabled successfully.")
                
                # Generate and print profiler statistics
                stats = pstats.Stats(profiler).sort_stats('cumulative')
                stats.print_stats()
                logging.info("Profiler statistics printed successfully.")
            except Exception as e:
                logging.error(f"Error handling profiler during resource cleanup: {e}")
        else:
            log_debug("No profiler actions required based on logging choice.")
    log_debug("Resource cleanup process completed.")

def get_logging_choice() -> str:
    """
    Prompt the user to select a logging level.

    Returns:
        str: The selected logging level ('V', 'B', or 'N').
    """
    logging_choice: str = input("Select logging level - Verbose (V), Brief (B), No Logging (N): ").strip().upper()
    log_debug(f"User selected logging choice: {logging_choice}")
    while logging_choice not in {'V', 'B', 'N'}:
        log_debug(f"Invalid logging choice: {logging_choice}")
        print("Invalid choice. Please enter 'V' for Verbose, 'B' for Brief, or 'N' for No Logging.")
        logging_choice = input("Select logging level - Verbose (V), Brief (B) or No Logging (N): ").strip().upper()
        log_debug(f"User re-selected logging choice: {logging_choice}")
    return logging_choice

def configure_logging(logging_choice: str) -> None:
    """
    Configure logging based on the user's choice.

    Parameters:
        logging_choice (str): The logging choice ('V', 'B', or 'N').
    """
    log_debug(f"Configuring logging with choice: {logging_choice}")
    if logging_choice == 'V':
        logging.basicConfig(level=logging.DEBUG)
        log_debug("Logging configured to DEBUG level.")
    elif logging_choice == 'B':
        logging.basicConfig(level=logging.INFO)
        log_debug("Logging configured to INFO level.")
    elif logging_choice == 'N':
        logging.disable(logging.CRITICAL)
        log_debug("Logging disabled.")

def initialize_profiler(logging_choice: str) -> cProfile.Profile:
    """
    Initialize the profiler based on the logging choice.

    Parameters:
        logging_choice (str): The logging choice ('V', 'B', or 'N').

    Returns:
        cProfile.Profile: The initialized profiler.
    """
    log_debug(f"Initializing profiler with logging choice: {logging_choice}")
    profiler: cProfile.Profile = cProfile.Profile()
    if logging_choice in {'V', 'B'}:
        profiler.enable()
        log_debug("Profiler enabled.")
    return profiler



def initialize_results_dataframe() -> pd.DataFrame:
    """
    Initialize the results DataFrame.

    Returns:
        pd.DataFrame: The initialized results DataFrame.
    """
    columns = [
        'Probabilities', 'H_X', 'H_Y', 'H_XY', 'P', 'E', 'Error Detection Rate',
        'Correction Capability', 'Adaptation Rate', 'Spatial Scale', 'Temporal Scale', 'Intelligence'
    ]
    log_debug(f"Initializing results DataFrame with columns: {columns}")
    return pd.DataFrame(columns=columns)

def update_results(results: pd.DataFrame, results_list: List[Dict[str, Any]], progress_bar: tqdm) -> None:
    """
    Update the results DataFrame and progress bar with the results from computations.

    Parameters:
        results (pd.DataFrame): The results DataFrame.
        results_list (List[Dict[str, Any]]): The list of results from computations.
        progress_bar (tqdm): The progress bar.
    """
    log_debug(f"Starting to update results DataFrame with {len(results_list)} new results.")
    for result in results_list:
        if result is not None:
            results = pd.concat([results, pd.DataFrame([result])], ignore_index=True)
            progress_bar.update(1)
            logging.info(f"Progress: {progress_bar.n}/{progress_bar.total} | Calculated Intelligence: {result['Intelligence']:.2f}")
    log_debug("Results DataFrame updated successfully.")

def main() -> None:
    """
    Main function to execute the model simulation, output results, and visualize the data with enhanced precision and detail using OpenCL for parallel computation.
    """
    try:
        # Setup OpenCL environment
        context, queue = CONTEXT, QUEUE
        log_debug("OpenCL context and queue initialized.")

        # Load parameter ranges
        parameter_ranges = load_parameter_ranges()
        log_debug(f"Parameter ranges loaded: {parameter_ranges.keys()}")

        # Calculate total operations
        total_operations = calculate_total_operations(parameter_ranges)
        log_debug(f"Total operations to be performed: {total_operations}")

        # Initialize progress bar
        progress_bar = initialize_progress_bar(total_operations)
        log_debug("Progress bar initialized.")

        # Initialize results DataFrame
        results = initialize_results_dataframe()
        log_debug("Results DataFrame initialized.")

        # Generate all combinations of parameters
        all_combinations = generate_combinations(parameter_ranges)
        log_debug(f"Generated {len(all_combinations)} combinations of parameters.")

        # Process each combination
        for idx, combination in enumerate(all_combinations):
            try:
                # Extract individual parameters
                probabilities = combination['probabilities_set']
                H_X = combination['H_X_set']
                H_Y = combination['H_Y_set']
                H_XY = combination['H_XY_set']
                P = combination['P_set']
                E = combination['E_set']
                error_detection_rate = combination['error_detection_rate_set']
                correction_capability = combination['correction_capability_set']
                adaptation_rate = combination['adaptation_rate_set']
                spatial_scale = combination['spatial_scale_set']
                temporal_scale = combination['temporal_scale_set']

                # Calculate metrics
                H_X_value = calculate_entropy(probabilities, alpha_parameters, context, queue)
                I_XY_value = calculate_mutual_information(H_X, H_Y, H_XY, alpha_parameters, context, queue)
                O_value = calculate_operational_efficiency(P, E, alpha_parameters, context, queue)
                Em_value = calculate_error_management(error_detection_rate, correction_capability, alpha_parameters, context, queue)
                A_value = calculate_adaptability(adaptation_rate, alpha_parameters, context, queue)
                Volume_value = calculate_volume(spatial_scale, alpha_parameters, context, queue)
                Time_value = calculate_time(temporal_scale, alpha_parameters, context, queue)

                # Compute intelligence metric
                I = alpha_parameters["k"] * (H_X_value * I_XY_value * O_value * Em_value * A_value) / (Volume_value * Time_value)

                # Update results DataFrame
                results = pd.concat([results, pd.DataFrame({
                    'Probabilities': [probabilities], 'H_X': [H_X], 'H_Y': [H_Y], 'H_XY': [H_XY],
                    'P': [P], 'E': [E], 'Error Detection Rate': [error_detection_rate],
                    'Correction Capability': [correction_capability], 'Adaptation Rate': [adaptation_rate],
                    'Spatial Scale': [spatial_scale], 'Temporal Scale': [temporal_scale], 'Intelligence': [I]
                })], ignore_index=True)

                # Update progress bar
                progress_bar.update(1)
                sys.stdout.write(f"\rmain: Progress: {idx+1}/{total_operations} | Last operation: Calculated Intelligence: {I:.2f}")
                sys.stdout.flush()

            except Exception as e:
                log_error(f"Error processing combination {idx}: {e}")

        # Close progress bar
        progress_bar.close()
        log_debug("Progress bar closed.")

        # Print and visualize results
        print(results.to_string(index=False))
        visualize_results(results)

    except Exception as e:
        log_error(f"Error in main execution: {e}")
    finally:
        # Cleanup resources
        cleanup_resources(context, queue, profiler, logging_choice)
        log_debug("Resources cleaned up.")

if __name__ == "__main__":
    # Get logging choice from user
    logging_choice = get_logging_choice()
    configure_logging(logging_choice)
    profiler = initialize_profiler(logging_choice)

    # Execute main function
    main()
