import numpy as np
import pandas as pd
import logging
from typing import Tuple, Dict, Any, List, Callable, Union
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joypy
from tqdm import tqdm
import pyopencl as cl
import sys
import cProfile
import pstats
from pytools.persistent_dict import PersistentDict
from threading import Lock, Thread, Event
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import multiprocessing
import time
import logging

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

try:
    log_debug("Setting up OpenCL environment.")
    CONTEXT, QUEUE = setup_opencl_environment()
    log_debug("OpenCL environment setup completed.")
except Exception as e:
    log_error(f"Failed to set up OpenCL environment: {e}")
    raise

GROUP_SIZE: int = 256
kernel_directory: str = "/home/lloyd/UniversalIntelligencePotential/kernels/"

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

def initialize_progress_bar(total_operations: int) -> Dict[str, int]:
    """
    Initialize the progress bar.
    
    Parameters:
        total_operations (int): The total number of operations.
    
    Returns:
        Dict[str, int]: The initialized progress bar.
    """
    log_debug(f"Initializing progress bar with total operations: {total_operations}")
    return {"total": total_operations, "current": 0}

def update_progress_bar(progress_bar: Dict[str, int]) -> None:
    """
    Update the progress bar.
    
    Parameters:
        progress_bar (Dict[str, int]): The progress bar object.
    """
    progress_bar["current"] += 1
    log_debug(f"Progress updated: {progress_bar['current']}/{progress_bar['total']}")
    print(f"Progress: {progress_bar['current']}/{progress_bar['total']}")

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


def load_parameter_ranges() -> Dict[str, Any]:
    """
    Load or define the parameter ranges for the model.
    
    Returns:
        Dict[str, Any]: A dictionary containing the parameter ranges.
    """
    if os.path.exists('parameter_ranges.npz'):
        with np.load('parameter_ranges.npz') as data:
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
        H_X_set = np.linspace(0.2, 2.0, 5).astype(int)  # Convert to int if used as indices
        H_Y_set = np.linspace(0.2, 2.0, 5).astype(int)
        H_XY_set = np.linspace(0.5, 3.0, 5).astype(int)
        P_set = np.linspace(100.0, 2000.0, 5).astype(int)
        E_set = np.linspace(20.0, 100.0, 5).astype(int)
        error_detection_rate_set = np.linspace(0.5, 1.0, 3).astype(int)
        correction_capability_set = np.linspace(0.5, 1.0, 3).astype(int)
        adaptation_rate_set = np.linspace(0.3, 1.0, 4).astype(int)
        spatial_scale_set = np.linspace(0.5, 1.5, 3).astype(int)
        temporal_scale_set = np.linspace(0.5, 1.5, 3).astype(int)

        np.savez('parameter_ranges.npz', probabilities_set=probabilities_set, H_X_set=H_X_set, H_Y_set=H_Y_set, H_XY_set=H_XY_set,
                 P_set=P_set, E_set=E_set, error_detection_rate_set=error_detection_rate_set,
                 correction_capability_set=correction_capability_set, adaptation_rate_set=adaptation_rate_set,
                 spatial_scale_set=spatial_scale_set, temporal_scale_set=temporal_scale_set)
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

def ensure_integer_indices(indices: Union[List[Any], np.ndarray]) -> List[int]:
    """
    Ensure that all indices are integers, converting if necessary.

    Parameters:
        indices (Union[List[Any], np.ndarray]): A list or array of indices which might not be of integer type.

    Returns:
        List[int]: A list of integer indices.
    """
    if isinstance(indices, np.ndarray):
        return indices.astype(int).tolist()
    return [int(index) for index in indices]

def process_data_using_indices(data: np.ndarray, indices: Union[List[Any], np.ndarray]) -> np.ndarray:
    """
    Process data using provided indices, ensuring indices are integers.

    Parameters:
        data (np.ndarray): The data to process.
        indices (Union[List[Any], np.ndarray]): Indices used for processing data.

    Returns:
        np.ndarray: Processed data.
    """
    valid_indices = ensure_integer_indices(indices)
    return data[valid_indices]

def validate_data(data: np.ndarray, name: str) -> None:
    """
    Validate the data before use.
    
    Parameters:
        data (np.ndarray): The data to validate.
        name (str): The name of the data set.
        
    Raises:
        ValueError: If the data is invalid.
    """
    log_debug(f"Validating data: {name}")
    if not isinstance(data, np.ndarray) or data.size == 0:
        log_error(f"{name} is invalid or empty.")
        raise ValueError(f"{name} is invalid or empty.")
    if np.any(data < 0):
        log_error(f"{name} contains negative values, which are not allowed.")
        raise ValueError(f"{name} contains negative values, which are not allowed.")
    log_debug(f"validate_data: {name} validation passed.")

def load_parameters_and_initialize() -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """
    Load parameter ranges and initialize the progress bar.
    
    This function loads various parameter ranges from a predefined function and initializes a progress bar
    based on the total number of operations derived from the product of the lengths of all parameter sets.
    
    Returns:
        Tuple[Dict[str, np.ndarray], Dict[str, int]]: A tuple containing the parameter ranges as a dictionary
        of numpy arrays, and an initialized progress bar object.
        
    Raises:
        TypeError: If the loaded parameter_ranges is not a dictionary.
        ValueError: If any required parameter set is missing or invalid.
    """
    log_debug("Loading parameter ranges and initializing progress bar.")
    parameter_ranges = load_parameter_ranges()
    validate_parameter_ranges(parameter_ranges)
    total_operations = calculate_total_operations(parameter_ranges)
    progress_bar = initialize_progress_bar(total_operations)
    log_debug("Parameter ranges loaded and progress bar initialized successfully.")
    return parameter_ranges, progress_bar

def validate_parameter_ranges(parameter_ranges: Dict[str, np.ndarray]) -> None:
    """
    Validate the loaded parameter ranges.
    
    Parameters:
        parameter_ranges (Dict[str, np.ndarray]): The loaded parameter ranges.
        
    Raises:
        TypeError: If parameter_ranges is not a dictionary.
        ValueError: If any required parameter set is missing or invalid.
    """
    log_debug("Validating parameter ranges.")
    if not isinstance(parameter_ranges, dict):
        log_error("Expected parameter_ranges to be a dictionary.")
        raise TypeError("Expected parameter_ranges to be a dictionary.")

    required_sets = [
        'H_X_set', 
        'H_Y_set', 
        'H_XY_set', 
        'P_set', 
        'E_set', 
        'error_detection_rate_set', 
        'correction_capability_set', 
        'adaptation_rate_set', 
        'spatial_scale_set', 
        'temporal_scale_set',
        'k_set'
    ]
    for key in required_sets:
        if key not in parameter_ranges or not isinstance(parameter_ranges[key], np.ndarray):
            log_error(f"Missing or invalid parameter set for {key}.")
            raise ValueError(f"Missing or invalid parameter set for {key}.")
        # Ensure all elements in the parameter sets are integers if they are used as indices
        if not np.issubdtype(parameter_ranges[key].dtype, np.integer):
            log_error(f"Parameter set {key} contains non-integer values, which are not allowed.")
            raise ValueError(f"Parameter set {key} contains non-integer values, which are not allowed.")
    log_debug("Parameter ranges validation passed.")

def calculate_total_operations(parameter_ranges: Dict[str, np.ndarray]) -> int:
    """
    Calculate the total number of operations based on the product of the sizes of all parameter sets.
    
    Parameters:
        parameter_ranges (Dict[str, np.ndarray]): The loaded parameter ranges.
        
    Returns:
        int: The total number of operations.
    """
    log_debug("Calculating total number of operations based on parameter ranges.")
    # Ensure all parameter sets are numpy arrays and convert their lengths to integers
    total_operations = 1
    for key, value in parameter_ranges.items():
        if not isinstance(value, np.ndarray):
            log_error(f"Expected parameter set {key} to be a numpy array.")
            raise TypeError(f"Expected parameter set {key} to be a numpy array.")
        total_operations *= int(len(value))
    log_debug(f"Total number of operations calculated: {total_operations}")
    return total_operations

def generate_combinations(parameter_ranges: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
    """
    Generate all possible combinations of parameter values.
    
    Parameters:
        parameter_ranges (Dict[str, np.ndarray]): A dictionary containing parameter ranges as numpy arrays.
        
    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each representing a unique combination of parameter values.
    """
    log_debug("Generating all possible combinations of parameter values.")
    keys = list(parameter_ranges.keys())
    values = list(parameter_ranges.values())
    combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
    log_debug(f"Generated {len(combinations)} combinations of parameter values.")
    return combinations

def validate_alpha_params(alpha_params: Dict[str, float]) -> None:
    """
    Validate the alpha parameters dictionary.
    
    Parameters:
        alpha_params (Dict[str, float]): The alpha parameters dictionary.
        
    Raises:
        TypeError: If alpha_params is not a dictionary.
        ValueError: If any value in alpha_params is not a float.
    """
    log_debug("Validating alpha parameters.")
    if not isinstance(alpha_params, dict):
        log_error("alpha_params must be a dictionary.")
        raise TypeError("alpha_params must be a dictionary.")
    if not all(isinstance(value, float) for value in alpha_params.values()):
        log_error("All values in alpha_params must be of type float.")
        raise ValueError("All values in alpha_params must be of type float.")
    log_debug("Alpha parameters validation passed.")

def prepare_alpha_values(alpha_params: Dict[str, float], keys: List[str]) -> np.ndarray:
    """
    Prepare the alpha values array from the alpha parameters dictionary.
    
    Parameters:
        alpha_params (Dict[str, float]): The alpha parameters dictionary.
        keys (List[str]): The keys to extract from the dictionary.
        
    Returns:
        np.ndarray: The prepared alpha values array.
    """
    log_debug(f"Preparing alpha values array from alpha parameters for keys: {keys}")
    # Ensure all keys are valid and convert indices to integers if necessary
    alpha_values = []
    for key in keys:
        value = alpha_params.get(key, 1.0)
        if not isinstance(value, (int, float)):
            log_error(f"Alpha parameter for key '{key}' must be an int or float, got {type(value)} instead.")
            raise ValueError(f"Alpha parameter for key '{key}' must be an int or float, got {type(value)} instead.")
        alpha_values.append(float(value))  # Ensure all values are floats
    
    # Convert the list to a numpy array of type float32
    alpha_values_array = np.array(alpha_values, dtype=np.float32)
    log_debug(f"Alpha values array prepared: {alpha_values_array}")
    return alpha_values_array

def create_buffers(context: cl.Context, data: np.ndarray, alpha_values: np.ndarray) -> Tuple[cl.Buffer, cl.Buffer, cl.Buffer]:
    """
    Create OpenCL buffers for data, alpha values, and result.
    
    Parameters:
        context (cl.Context): The OpenCL context.
        data (np.ndarray): The input data array.
        alpha_values (np.ndarray): The alpha values array.
        
    Returns:
        Tuple[cl.Buffer, cl.Buffer, cl.Buffer]: A tuple containing the data buffer, alpha buffer, and result buffer.
    """
    log_debug("Creating OpenCL buffers for data, alpha values, and result.")
    data_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=data)
    alpha_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=alpha_values)
    result_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, data.nbytes)
    log_debug("OpenCL buffers created successfully.")
    return data_buf, alpha_buf, result_buf

def execute_kernel(queue: cl.CommandQueue, kernel_func: Callable, buffers: Tuple[cl.Buffer, cl.Buffer, cl.Buffer], global_size: Tuple[int]) -> None:
    """
    Execute the provided kernel function with the given buffers and global size.
    
    Parameters:
        queue (cl.CommandQueue): The OpenCL command queue.
        kernel_func (Callable): The kernel function to execute.
        buffers (Tuple[cl.Buffer, cl.Buffer, cl.Buffer]): A tuple of OpenCL buffers.
        global_size (Tuple[int]): The global size for the kernel execution.
    """
    log_debug(f"Executing kernel function {kernel_func} with global size {global_size} and buffers {buffers}.")
    try:
        kernel_func(queue, global_size, None, *buffers)
        log_debug(f"Kernel function {kernel_func} executed successfully.")
    except Exception as e:
        log_error(f"Error executing kernel function {kernel_func}: {e}")
        raise

def retrieve_result(queue: cl.CommandQueue, result_buf: cl.Buffer, size: int) -> float:
    """
    Retrieve the result from the result buffer and return the sum.
    
    Parameters:
        queue (cl.CommandQueue): The OpenCL command queue.
        result_buf (cl.Buffer): The result buffer.
        size (int): The size of the result array.
        
    Returns:
        float: The sum of the result array.
    """
    log_debug(f"Retrieving result from buffer {result_buf} with size {size}.")
    try:
        result = np.empty(size, dtype=np.float32)
        cl.enqueue_copy(queue, result, result_buf).wait()
        result_sum = np.sum(result)
        log_debug(f"Result retrieved and summed: {result_sum}")
        return result_sum
    except Exception as e:
        log_error(f"Error retrieving result from buffer {result_buf}: {e}")
        raise

def release_buffers(*buffers: cl.Buffer) -> None:
    """
    Release the provided OpenCL buffers.
    
    Parameters:
        *buffers (cl.Buffer): The buffers to release.
    """
    log_debug(f"Releasing buffers: {buffers}")
    try:
        for buf in buffers:
            buf.release()
            log_debug(f"Buffer {buf} released successfully.")
    except Exception as e:
        log_error(f"Error releasing buffers: {e}")
        raise

def calculate_entropy(probabilities: np.ndarray, alpha_params: Dict[str, float], context: cl.Context, queue: cl.CommandQueue) -> float:
    """
    Calculate the Shannon entropy of a probability distribution using scaling factors and OpenCL for parallel processing.
    
    Parameters:
        probabilities (np.ndarray): The probability distribution array.
        alpha_params (Dict[str, float]): Dictionary of alpha parameters for scaling.
        context (cl.Context): The OpenCL context.
        queue (cl.CommandQueue): The OpenCL command queue.
    
    Returns:
        float: The calculated Shannon entropy.
    
    Raises:
        cl.Error: If an OpenCL error occurs during the calculation.
        ValueError: If a validation error occurs.
    """
    log_debug(f"Starting calculation of Shannon entropy with probabilities: {probabilities} and alpha_params: {alpha_params}")
    try:
        validate_data(probabilities, 'probabilities')
        validate_alpha_params(alpha_params)

        alpha_values = prepare_alpha_values(alpha_params, ["alpha_H", "alpha_Pi", "alpha_log"])
        log_debug(f"Alpha values prepared: {alpha_values}")

        data_buf, alpha_buf, result_buf = create_buffers(context, probabilities, alpha_values)
        log_debug(f"Buffers created: data_buf={data_buf}, alpha_buf={alpha_buf}, result_buf={result_buf}")

        execute_kernel(queue, entropy_program.calculate_entropy, (data_buf, alpha_buf, result_buf), global_size=(probabilities.size,))

        entropy = retrieve_result(queue, result_buf, probabilities.size)
        log_debug(f"Calculated Shannon entropy: {entropy}")
        return entropy
    except (cl.Error, ValueError) as e:
        log_error(f"Error calculating Shannon entropy: {e}")
        raise
    finally:
        release_buffers(data_buf, alpha_buf, result_buf)

def calculate_mutual_information(H_X: float, H_Y: float, H_XY: float, alpha_params: Dict[str, float], context: cl.Context, queue: cl.CommandQueue) -> float:
    """
    Calculate the mutual information based on entropies of X, Y, and their joint distribution using OpenCL for parallel processing.
    
    Parameters:
        H_X (float): Entropy of X.
        H_Y (float): Entropy of Y.
        H_XY (float): Joint entropy of X and Y.
        alpha_params (Dict[str, float]): Dictionary of alpha parameters for scaling.
        context (cl.Context): The OpenCL context.
        queue (cl.CommandQueue): The OpenCL command queue.
    
    Returns:
        float: The calculated mutual information.
    
    Raises:
        cl.Error: If an OpenCL error occurs during the calculation.
        ValueError: If a validation error occurs.
    """
    log_debug(f"Starting calculation of mutual information with H_X: {H_X}, H_Y: {H_Y}, H_XY: {H_XY}, alpha_params: {alpha_params}")
    try:
        validate_alpha_params(alpha_params)
        validate_data(np.array([H_X, H_Y, H_XY]), 'entropies')

        alpha_values = prepare_alpha_values(alpha_params, ["alpha_I", "alpha_HX", "alpha_HY", "alpha_HXY"])
        log_debug(f"Alpha values prepared: {alpha_values}")

        buffers = create_buffers(context, np.array([H_X, H_Y, H_XY], dtype=np.float32), alpha_values)
        log_debug(f"Buffers created: {buffers}")

        execute_kernel(queue, mutual_information_program.calculate_mutual_information, buffers, global_size=(1,))

        mutual_info = retrieve_result(queue, buffers[-1], 1)
        log_debug(f"Calculated mutual information: {mutual_info}")
        return mutual_info
    except (cl.Error, ValueError) as e:
        log_error(f"Error calculating mutual information: {e}")
        raise
    finally:
        release_buffers(*buffers)

def calculate_operational_efficiency(P: float, E: float, alpha_params: Dict[str, float], context: cl.Context, queue: cl.CommandQueue) -> float:
    """
    Calculate the operational efficiency based on performance and energy consumption using OpenCL for parallel processing.
    
    Parameters:
        P (float): Performance measure, typically computational power or output rate.
        E (float): Energy consumption measure.
        alpha_params (Dict[str, float]): Dictionary of alpha parameters for scaling.
        context (cl.Context): The OpenCL context.
        queue (cl.CommandQueue): The OpenCL command queue.
        
    Returns:
        float: The calculated operational efficiency.
        
    Raises:
        cl.Error: If an OpenCL error occurs during the calculation.
        ValueError: If a validation error occurs.
    """
    log_debug(f"Starting calculation of operational efficiency with P: {P}, E: {E}, alpha_params: {alpha_params}")
    try:
        validate_data(np.array([P, E]), 'performance_energy')
        validate_alpha_params(alpha_params)

        alpha_values = prepare_alpha_values(alpha_params, ["alpha_O", "alpha_P", "alpha_E"])
        log_debug(f"Alpha values prepared: {alpha_values}")

        buffers = create_buffers(context, np.array([P, E], dtype=np.float32), alpha_values)
        log_debug(f"Buffers created: {buffers}")

        execute_kernel(queue, operational_efficiency_program.calculate_operational_efficiency, buffers, global_size=(1,))

        efficiency = retrieve_result(queue, buffers[-1], 1)
        log_debug(f"Calculated operational efficiency: {efficiency}")
        return efficiency
    except (cl.Error, ValueError) as e:
        log_error(f"Error calculating operational efficiency: {e}")
        raise
    finally:
        release_buffers(*buffers)

def calculate_error_management(error_detection_rate: float, correction_capability: float, alpha_params: Dict[str, float], context: cl.Context, queue: cl.CommandQueue) -> float:
    """
    Calculate the error management effectiveness based on error detection rate and correction capability using OpenCL for parallel processing.
    
    Parameters:
        error_detection_rate (float): The rate at which errors are detected.
        correction_capability (float): The capability to correct detected errors.
        alpha_params (Dict[str, float]): Dictionary of alpha parameters for scaling.
        context (cl.Context): The OpenCL context.
        queue (cl.CommandQueue): The OpenCL command queue.
        
    Returns:
        float: The calculated error management effectiveness.
        
    Raises:
        cl.Error: If an OpenCL error occurs during the calculation.
        ValueError: If a validation error occurs.
    """
    log_debug(f"Starting calculation of error management with error_detection_rate: {error_detection_rate}, correction_capability: {correction_capability}, alpha_params: {alpha_params}")
    try:
        validate_data(np.array([error_detection_rate, correction_capability]), 'error_management_params')
        validate_alpha_params(alpha_params)

        alpha_values = prepare_alpha_values(alpha_params, ["alpha_Em", "alpha_Error_Detection", "alpha_Correction"])
        log_debug(f"Alpha values prepared: {alpha_values}")

        buffers = create_buffers(context, np.array([error_detection_rate, correction_capability], dtype=np.float32), alpha_values)
        log_debug(f"Buffers created: {buffers}")

        execute_kernel(queue, error_management_program.calculate_error_management, buffers, global_size=(1,))

        error_management_value = retrieve_result(queue, buffers[-1], 1)
        log_debug(f"Calculated error management effectiveness: {error_management_value}")
        return error_management_value
    except (cl.Error, ValueError) as e:
        log_error(f"Error calculating error management effectiveness: {e}")
        raise
    finally:
        release_buffers(*buffers)

def calculate_adaptability(adaptation_rate: float, alpha_params: Dict[str, float], context: cl.Context, queue: cl.CommandQueue) -> float:
    """
    Calculate the adaptability based on the adaptation rate using OpenCL for parallel processing.
    
    Parameters:
        adaptation_rate (float): The rate at which the system can adapt to changes.
        alpha_params (Dict[str, float]): Dictionary of alpha parameters for scaling.
        context (cl.Context): The OpenCL context.
        queue (cl.CommandQueue): The OpenCL command queue.
        
    Returns:
        float: The calculated adaptability.
        
    Raises:
        cl.Error: If an OpenCL error occurs during the calculation.
        ValueError: If a validation error occurs.
    """
    log_debug(f"Starting calculation of adaptability with adaptation_rate: {adaptation_rate}, alpha_params: {alpha_params}")
    try:
        validate_data(np.array([adaptation_rate]), 'adaptation_rate')
        validate_alpha_params(alpha_params)

        alpha_values = prepare_alpha_values(alpha_params, ["alpha_A", "alpha_Adaptation_Rate"])
        log_debug(f"Alpha values prepared: {alpha_values}")

        buffers = create_buffers(context, np.array([adaptation_rate], dtype=np.float32), alpha_values)
        log_debug(f"Buffers created: {buffers}")

        execute_kernel(queue, adaptability_program.calculate_adaptability, buffers, global_size=(1,))

        adaptability_value = retrieve_result(queue, buffers[-1], 1)
        log_debug(f"Calculated adaptability: {adaptability_value}")
        return adaptability_value
    except (cl.Error, ValueError) as e:
        log_error(f"Error calculating adaptability: {e}")
        raise
    finally:
        release_buffers(*buffers)

def calculate_volume(spatial_scale: float, alpha_params: Dict[str, float], context: cl.Context, queue: cl.CommandQueue) -> float:
    """
    Calculate the volume based on the spatial scale using OpenCL for parallel processing.
    
    Parameters:
        spatial_scale (float): The spatial scale factor, typically representing physical dimensions.
        alpha_params (Dict[str, float]): Dictionary of alpha parameters for scaling.
        context (cl.Context): The OpenCL context.
        queue (cl.CommandQueue): The OpenCL command queue.
        
    Returns:
        float: The calculated volume.
        
    Raises:
        cl.Error: If an OpenCL error occurs during the calculation.
        ValueError: If a validation error occurs.
    """
    log_debug(f"Starting calculation of volume with spatial_scale: {spatial_scale}, alpha_params: {alpha_params}")
    try:
        validate_data(np.array([spatial_scale]), 'spatial_scale')
        validate_alpha_params(alpha_params)

        alpha_values = prepare_alpha_values(alpha_params, ["alpha_Volume", "alpha_Spatial_Scale"])
        log_debug(f"Alpha values prepared: {alpha_values}")

        buffers = create_buffers(context, np.array([spatial_scale], dtype=np.float32), alpha_values)
        log_debug(f"Buffers created: {buffers}")

        execute_kernel(queue, volume_program.calculate_volume, buffers, global_size=(1,))

        volume_value = retrieve_result(queue, buffers[-1], 1)
        log_debug(f"Calculated volume: {volume_value}")
        return volume_value
    except (cl.Error, ValueError) as e:
        log_error(f"Error calculating volume: {e}")
        raise
    finally:
        release_buffers(*buffers)

def calculate_time(temporal_scale: float, alpha_params: Dict[str, float], context: cl.Context, queue: cl.CommandQueue) -> float:
    """
    Calculate the time based on the temporal scale using OpenCL for parallel processing.
    
    Parameters:
        temporal_scale (float): The temporal scale factor, typically representing time dimensions.
        alpha_params (Dict[str, float]): Dictionary of alpha parameters for scaling.
        context (cl.Context): The OpenCL context.
        queue (cl.CommandQueue): The OpenCL command queue.
        
    Returns:
        float: The calculated time.
        
    Raises:
        cl.Error: If an OpenCL error occurs during the calculation.
        ValueError: If a validation error occurs.
    """
    log_debug(f"Starting calculation of time with temporal_scale: {temporal_scale}, alpha_params: {alpha_params}")
    try:
        validate_data(np.array([temporal_scale]), 'temporal_scale')
        validate_alpha_params(alpha_params)

        alpha_values = prepare_alpha_values(alpha_params, ["alpha_t", "alpha_Temporal_Scale"])
        log_debug(f"Alpha values prepared: {alpha_values}")

        buffers = create_buffers(context, np.array([temporal_scale], dtype=np.float32), alpha_values)
        log_debug(f"Buffers created: {buffers}")

        execute_kernel(queue, time_program.calculate_time, buffers, global_size=(1,))

        time_value = retrieve_result(queue, buffers[-1], 1)
        log_debug(f"Calculated time: {time_value}")
        return time_value
    except (cl.Error, ValueError) as e:
        log_error(f"Error calculating time: {e}")
        raise
    finally:
        release_buffers(*buffers)

def compute_intelligence(combination: Dict[str, float], context: cl.Context, queue: cl.CommandQueue) -> float:
    """
    Compute the intelligence metric for a given parameter combination, handling exceptions and ensuring robust computation.
    
    Parameters:
        combination (Dict[str, float]): The parameter combination.
        context (cl.Context): The OpenCL context.
        queue (cl.CommandQueue): The OpenCL command queue.
    
    Returns:
        float: The computed intelligence metric or None if an error occurs.
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

def calculate_total_operations(parameter_ranges: Dict[str, Dict[str, np.ndarray]]) -> int:
    """
    Calculate the total number of operations based on parameter ranges.

    Parameters:
        parameter_ranges (Dict[str, Dict[str, np.ndarray]]): The parameter ranges.

    Returns:
        int: The total number of operations.
    """
    total_operations = int(np.prod([len(param['values']) for param in parameter_ranges.values()]))
    log_debug(f"Total operations calculated: {total_operations}")
    return total_operations

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

def execute_computations(all_combinations: List[Dict[str, Any]], context: cl.Context, queue: cl.CommandQueue, max_workers: int) -> List[Dict[str, Any]]:
    """
    Execute computations in parallel using a ThreadPoolExecutor.

    Parameters:
        all_combinations (List[Dict[str, Any]]): All combinations of parameters.
        context (cl.Context): The OpenCL context.
        queue (cl.CommandQueue): The OpenCL command queue.
        max_workers (int): The maximum number of workers.

    Returns:
        List[Dict[str, Any]]: The list of results from computations.
    """
    log_debug(f"Starting parallel execution of computations with {max_workers} workers.")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(lambda comb: compute_intelligence(comb, context, queue), all_combinations))
    log_debug(f"Parallel execution completed with {len(results)} results.")
    return results

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
    This function is designed to be highly modular, scalable, and robust, incorporating detailed logging and error handling to ensure maximum efficiency and efficacy.
    The function configures logging based on user input, initializes computation resources, loads parameters, computes results in parallel, and visualizes them.
    It also handles exceptions and cleans up resources meticulously, ensuring that all operations are logged and monitored.
    """
    log_debug("Starting main function.")
    try:
        # Initialize OpenCL context and command queue
        log_debug("Attempting to initialize OpenCL context and command queue.")
        context: cl.Context = CONTEXT
        queue: cl.CommandQueue = QUEUE
        log_debug("OpenCL context and command queue initialized successfully.")

        # Configure logging and profiler
        log_debug("Configuring logging and profiler based on user input.")
        logging_choice: str = get_logging_choice()
        log_debug(f"User selected logging choice: {logging_choice}")
        configure_logging(logging_choice)
        profiler: cProfile.Profile = initialize_profiler(logging_choice)
        log_debug("Logging and profiler configured successfully.")

        # Load parameters and initialize resources
        log_debug("Loading parameter ranges.")
        parameter_ranges: Dict[str, Dict[str, np.ndarray]] = load_parameter_ranges()
        log_debug(f"Parameter ranges loaded successfully: {parameter_ranges}")
        total_operations: int = calculate_total_operations(parameter_ranges)
        log_debug(f"Total operations calculated: {total_operations}")
        progress_bar: tqdm = initialize_progress_bar(total_operations)
        log_debug("Progress bar initialized successfully.")
        results: pd.DataFrame = initialize_results_dataframe()
        log_debug("Results DataFrame initialized successfully.")

        # Generate all parameter combinations
        log_debug("Generating all parameter combinations.")
        all_combinations: List[Dict[str, Any]] = generate_combinations(parameter_ranges)
        log_debug(f"Generated {len(all_combinations)} parameter combinations successfully.")
        max_workers: int = multiprocessing.cpu_count()
        log_debug(f"Maximum number of workers determined: {max_workers}")

        # Execute computations in parallel
        log_debug("Starting parallel execution of computations.")
        results_list: List[Dict[str, Any]] = execute_computations(all_combinations, context, queue, max_workers)
        log_debug(f"Parallel execution completed. Number of results obtained: {len(results_list)}")
        update_results(results, results_list, progress_bar)
        log_debug("Results DataFrame updated successfully with computed results.")

        # Close progress bar and log completion
        log_debug("Closing progress bar.")
        progress_bar.close()
        log_info("All combinations have been processed successfully.")
        
        # Visualize results
        log_debug("Starting visualization of results.")
        visualize_results(results)
        log_debug("Results visualization completed successfully.")

    except KeyboardInterrupt:
        log_error("Process interrupted by user.")
    except Exception as e:
        log_error(f"An error occurred during execution: {e}")
    finally:
        log_debug("Starting cleanup of resources.")
        cleanup_resources(context, queue, profiler, logging_choice)
        log_debug("Resource cleanup completed successfully.")

if __name__ == "__main__":
    log_debug("Executing main function as script entry point.")
    main()
    log_debug("Main function execution completed.")
