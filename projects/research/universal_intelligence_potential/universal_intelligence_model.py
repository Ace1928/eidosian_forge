import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Dict, Any, Optional, Union
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joypy  # Library for ridge plots
from mpl_toolkits.mplot3d import Axes3D
import time
from datetime import datetime
from tqdm import tqdm
import multiprocessing as mp
import pyopencl as cl
import pyopencl.array as cl_array
import sys
from logging.handlers import RotatingFileHandler
import bokeh
from bokeh.io import save, show
from bokeh.resources import INLINE

# Set environment variables for optimal OpenCL performance and debugging
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'  # Enable verbose output from the OpenCL compiler
os.environ['PYOPENCL_CTX'] = '0'  # Select the first available OpenCL context
os.environ['PYOPENCL_NO_CACHE'] = '1'  # Disable caching to ensure the latest kernel code is always used
os.environ['PYOPENCL_WAIT_FOR_KERNEL_COMPILATION'] = '1'  # Wait for kernel compilation to complete before execution

# Ensure the logs directory exists
log_directory = '/home/lloyd/UniversalIntelligencePotential/logs'
os.makedirs(log_directory, exist_ok=True)

# Configure advanced logging
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# General log handler
log_handler = RotatingFileHandler(os.path.join(log_directory, 'universal_intelligence_model.log'), 
                                  maxBytes=10*1024*1024, backupCount=50)
log_handler.setFormatter(log_formatter)
log_handler.setLevel(logging.DEBUG)

# Console log handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)

# Error log handler
error_log_handler = RotatingFileHandler(os.path.join(log_directory, 'universal_intelligence_model_error.log'), 
                                        maxBytes=10*1024*1024, backupCount=50)
error_log_handler.setFormatter(log_formatter)
error_log_handler.setLevel(logging.ERROR)

# Configure the root logger
logging.basicConfig(level=logging.DEBUG, handlers=[log_handler, console_handler, error_log_handler])

# Log the environment settings for verification
logging.info("Environment variables set for OpenCL:")
for key in ['PYOPENCL_COMPILER_OUTPUT', 'PYOPENCL_CTX', 'PYOPENCL_NO_CACHE', 'PYOPENCL_WAIT_FOR_KERNEL_COMPILATION']:
    logging.info(f"{key} = {os.environ[key]}")

# Setup OpenCL environment with robust device selection and resource utilization
def setup_opencl_environment():
    platforms = cl.get_platforms()
    devices = []
    for platform in platforms:
        devices.extend(platform.get_devices())

    if not devices:
        raise RuntimeError("No OpenCL devices found.")

    # Select the best device based on maximum compute units and global memory size
    best_device = max(devices, key=lambda d: (d.max_compute_units, d.global_mem_size))
    context = cl.Context([best_device])
    queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)
    
    logging.info(f"Selected OpenCL device: {best_device.name}")
    logging.debug(f"Device details: {best_device}")
    
    return context, queue, best_device

context, queue, device = setup_opencl_environment()

# Define the parameters for the model with explicit type annotations
alpha_parameters: Dict[str, float] = {
    "k": 1.0, "alpha_H": 1.0, "alpha_I": 1.0, "alpha_O": 1.0, "alpha_Em": 1.0,
    "alpha_A": 1.0, "alpha_Volume": 1.0, "alpha_t": 1.0, "alpha_Pi": 1.0,
    "alpha_log": 1.0, "alpha_HX": 1.0, "alpha_HY": 1.0, "alpha_HXY": 1.0,
    "alpha_P": 1.0, "alpha_E": 1.0, "alpha_Error_Detection": 1.0,
    "alpha_Correction": 1.0, "alpha_Adaptation_Rate": 1.0, "alpha_Spatial_Scale": 1.0,
    "alpha_Temporal_Scale": 1.0
}

# Kernel Codes
"""
This block of the code contains the OpenCL kernel codes as strings to keep everything in one file for easy maintenance and portability.
"""
# OpenCL kernel for calculating entropy
entropy_kernel_code = """
__kernel void calculate_entropy(__global const float *probabilities, __global const float *alpha_params, __global float *results, const int size) {
    int i = get_global_id(0);
    if (i < size) {
        // Load alpha parameters into local variables for faster access
        const float alpha_H = alpha_params[0];
        const float alpha_Pi = alpha_params[1];
        const float alpha_log = alpha_params[2];
        
        // Read the probability value
        const float prob = probabilities[i];
        
        // Ensure probability is non-zero to avoid log(0)
        if (prob > 0.0f) {
            // Calculate entropy component
            results[i] = alpha_H * (-prob * log(alpha_log * prob) * alpha_Pi);
        } else {
            results[i] = 0.0f;  // Assign zero if probability is zero
        }
    }
}
"""
entropy_program = cl.Program(context, entropy_kernel_code).build(options='-cl-fast-relaxed-math -cl-mad-enable')

# OpenCL kernel for calculating mutual information
mutual_information_kernel_code = """
__kernel void calculate_mutual_information(
    __global const float *H_X, 
    __global const float *H_Y, 
    __global const float *H_XY, 
    __global const float *alpha_params, 
    __global float *result
) {
    int i = get_global_id(0);
    if (i == 0) {
        // Load alpha parameters into local variables for faster access
        const float alpha_I = alpha_params[0];
        const float alpha_HX = alpha_params[1];
        const float alpha_HY = alpha_params[2];
        const float alpha_HXY = alpha_params[3];

        // Calculate mutual information
        const float mutual_information = alpha_I * (
            alpha_HX * H_X[0] + 
            alpha_HY * H_Y[0] - 
            alpha_HXY * H_XY[0]
        );

        // Store the result
        result[0] = mutual_information;
    }
}
"""
mutual_information_program = cl.Program(context, mutual_information_kernel_code).build(options='-cl-fast-relaxed-math -cl-mad-enable')

# OpenCL kernel for calculating operational efficiency
operational_efficiency_kernel_code = """
__kernel void calculate_operational_efficiency(
    __global const float *P, 
    __global const float *E, 
    __global const float *alpha_params, 
    __global float *result
) {
    int i = get_global_id(0);
    if (i == 0) {
        // Load alpha parameters into local variables for faster access
        const float alpha_O = alpha_params[0];
        const float alpha_P = alpha_params[1];
        const float alpha_E = alpha_params[2];

        // Read the input values
        const float P_value = P[0];
        const float E_value = E[0];

        // Calculate operational efficiency, ensuring no division by zero
        if (E_value != 0.0f) {
            result[0] = alpha_O * (alpha_P * P_value / (alpha_E * E_value));
        } else {
            result[0] = 0.0f;  // Assign zero if E_value is zero to avoid division by zero
        }
    }
}
"""
operational_efficiency_program = cl.Program(context, operational_efficiency_kernel_code).build(options='-cl-fast-relaxed-math -cl-mad-enable')

# OpenCL kernel for calculating error management
error_management_kernel_code = """
__kernel void calculate_error_management(
    __global const float *error_detection_rate, 
    __global const float *correction_capability, 
    __global const float *alpha_params, 
    __global float *result
) {
    // Get the global ID of the current work item
    int i = get_global_id(0);

    // Ensure only the first work item performs the calculation
    if (i == 0) {
        // Load alpha parameters into local variables for faster access
        const float alpha_Em = alpha_params[0];
        const float alpha_Error_Detection = alpha_params[1];
        const float alpha_Correction = alpha_params[2];

        // Read the input values
        const float error_detection_rate_value = error_detection_rate[0];
        const float correction_capability_value = correction_capability[0];

        // Calculate error management metric
        const float error_management = alpha_Em * (
            alpha_Error_Detection * error_detection_rate_value * 
            alpha_Correction * correction_capability_value
        );

        // Store the result
        result[0] = error_management;
    }
}
"""
error_management_program = cl.Program(context, error_management_kernel_code).build(options='-cl-fast-relaxed-math -cl-mad-enable')

# OpenCL kernel for calculating adaptability
adaptability_kernel_code = """
__kernel void calculate_adaptability(
    __global const float *adaptation_rate, 
    __global const float *alpha_params, 
    __global float *result
) {
    // Get the global ID of the current work item
    int i = get_global_id(0);

    // Ensure only the first work item performs the calculation
    if (i == 0) {
        // Load alpha parameters into local variables for faster access
        const float alpha_A = alpha_params[0];
        const float alpha_Adaptation_Rate = alpha_params[1];

        // Read the input values
        const float adaptation_rate_value = adaptation_rate[0];

        // Calculate adaptability metric
        const float adaptability = alpha_A * (alpha_Adaptation_Rate * adaptation_rate_value);

        // Store the result
        result[0] = adaptability;
    }
}
"""
adaptability_program = cl.Program(context, adaptability_kernel_code).build(options='-cl-fast-relaxed-math -cl-mad-enable')

# OpenCL kernel for calculating volume
volume_kernel_code = """
__kernel void calculate_volume(
    __global const float *spatial_scale, 
    __global const float *alpha_params, 
    __global float *result
) {
    // Get the global ID of the current work item
    int i = get_global_id(0);

    // Ensure only the first work item performs the calculation
    if (i == 0) {
        // Load alpha parameters into local variables for faster access
        const float alpha_Volume = alpha_params[0];
        const float alpha_Spatial_Scale = alpha_params[1];

        // Read the input values
        const float spatial_scale_value = spatial_scale[0];

        // Calculate volume metric
        const float volume = alpha_Volume * (alpha_Spatial_Scale * spatial_scale_value);

        // Store the result
        result[0] = volume;
    }
}
"""
volume_program = cl.Program(context, volume_kernel_code).build(options='-cl-fast-relaxed-math -cl-mad-enable')

# OpenCL kernel for calculating time
time_kernel_code = """
__kernel void calculate_time(
    __global const float *temporal_scale, 
    __global const float *alpha_params, 
    __global float *result
) {
    // Get the global ID of the current work item
    int i = get_global_id(0);

    // Ensure only the first work item performs the calculation
    if (i == 0) {
        // Load alpha parameters into local variables for faster access
        const float alpha_t = alpha_params[0];
        const float alpha_Temporal_Scale = alpha_params[1];

        // Read the input values
        const float temporal_scale_value = temporal_scale[0];

        // Calculate time metric
        const float time_metric = alpha_t * (alpha_Temporal_Scale * temporal_scale_value);

        // Store the result
        result[0] = time_metric;
    }
}
"""
time_program = cl.Program(context, time_kernel_code).build(options='-cl-fast-relaxed-math -cl-mad-enable')

# Function to create an OpenCL buffer with robust error handling and exponential backoff
def create_buffer(context: cl.Context, size: int, data: np.ndarray = None) -> cl.Buffer:
    """
    Create an OpenCL buffer with robust error handling and exponential backoff.

    Args:
        context (cl.Context): The OpenCL context.
        size (int): The size of the buffer to create.
        data (np.ndarray, optional): The data to initialize the buffer with. Defaults to None.

    Returns:
        cl.Buffer: The created OpenCL buffer.

    Raises:
        MemoryError: If the buffer cannot be allocated after several attempts.
    """
    max_attempts: int = 5
    backoff_time: float = 1.0  # initial backoff time in seconds

    for attempt in range(max_attempts):
        try:
            if data is not None:
                buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=data)
            else:
                buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE, size)
            logging.debug(f"Buffer allocated successfully on attempt {attempt + 1}")
            logging.debug(f"Buffer details: context={context}, size={size}, data={data}")
            return buffer
        except cl.MemoryError as e:
            logging.warning(f"Failed to allocate buffer on attempt {attempt + 1}: {e}")
            logging.debug(f"Buffer allocation attempt {attempt + 1} failed with MemoryError: {e}")
            time.sleep(backoff_time)
            backoff_time *= 2  # exponential backoff
        except Exception as e:
            logging.error(f"Unexpected error on attempt {attempt + 1}: {e}")
            logging.debug(f"Unexpected error details: {e}", exc_info=True)
            raise

    raise MemoryError("Failed to allocate GPU buffer after several attempts.")

# Function to calculate Shannon entropy. The Shannon Entropy is a measure of the uncertainty or randomness of a set of probabilities.
def calculate_entropy(probabilities: np.ndarray, alpha_params: Dict[str, float]) -> float:
    prob_buf = alpha_buf = result_buf = None
    try:
        alpha_values = np.array([alpha_params["alpha_H"], alpha_params["alpha_Pi"], alpha_params["alpha_log"]], dtype=np.float32)
        prob_buf = create_buffer(context, probabilities.nbytes, probabilities)
        alpha_buf = create_buffer(context, alpha_values.nbytes, alpha_values)
        result_buf = create_buffer(context, probabilities.nbytes)
        entropy_program.calculate_entropy(queue, probabilities.shape, None, prob_buf, alpha_buf, result_buf, np.int32(len(probabilities)))
        result = np.empty_like(probabilities)
        cl.enqueue_copy(queue, result, result_buf).wait()
        entropy = np.sum(result)
        logging.debug(f"calculate_entropy: Calculated Shannon entropy: {entropy}")
        logging.debug(f"Probabilities: {probabilities}, Alpha Params: {alpha_params}, Result: {result}")
        return entropy
    except cl.MemoryError as e:
        logging.error(f"calculate_entropy: MemoryError calculating Shannon entropy: {e}")
        logging.debug(f"MemoryError details: {e}", exc_info=True)
        raise
    except Exception as e:
        logging.error(f"calculate_entropy: Error calculating Shannon entropy: {e}")
        logging.debug(f"Error details: {e}", exc_info=True)
        raise
    finally:
        if prob_buf: prob_buf.release()
        if alpha_buf: alpha_buf.release()
        if result_buf: result_buf.release()

# Function to calculate mutual information. The Mutual Information is a measure of the dependence between two random variables in the system.
def calculate_mutual_information(H_X: float, H_Y: float, H_XY: float, alpha_params: Dict[str, float]) -> float:
    H_X_buf = H_Y_buf = H_XY_buf = alpha_buf = result_buf = None
    try:
        alpha_values = np.array([alpha_params["alpha_I"], alpha_params["alpha_HX"], alpha_params["alpha_HY"], alpha_params["alpha_HXY"]], dtype=np.float32)
        H_X_buf = create_buffer(context, np.float32(0).nbytes, np.array([H_X], dtype=np.float32))
        H_Y_buf = create_buffer(context, np.float32(0).nbytes, np.array([H_Y], dtype=np.float32))
        H_XY_buf = create_buffer(context, np.float32(0).nbytes, np.array([H_XY], dtype=np.float32))
        alpha_buf = create_buffer(context, alpha_values.nbytes, alpha_values)
        result_buf = create_buffer(context, np.float32(0).nbytes)
        mutual_information_program.calculate_mutual_information(queue, (1,), None, H_X_buf, H_Y_buf, H_XY_buf, alpha_buf, result_buf)
        result = np.empty(1, dtype=np.float32)
        cl.enqueue_copy(queue, result, result_buf).wait()
        mutual_info = result[0]
        logging.debug(f"calculate_mutual_information: Calculated mutual information: {mutual_info}")
        logging.debug(f"H_X: {H_X}, H_Y: {H_Y}, H_XY: {H_XY}, Alpha Params: {alpha_params}, Result: {result}")
        return mutual_info
    except cl.MemoryError as e:
        logging.error(f"calculate_mutual_information: MemoryError calculating mutual information: {e}")
        logging.debug(f"MemoryError details: {e}", exc_info=True)
        raise
    except Exception as e:
        logging.error(f"calculate_mutual_information: Error calculating mutual information: {e}")
        logging.debug(f"Error details: {e}", exc_info=True)
        raise
    finally:
        if H_X_buf: H_X_buf.release()
        if H_Y_buf: H_Y_buf.release()
        if H_XY_buf: H_XY_buf.release()
        if alpha_buf: alpha_buf.release()
        if result_buf: result_buf.release()

# Function to calculate operational efficiency (essentially the ability of the system to process information in a timely manner)
def calculate_operational_efficiency(P: float, E: float, alpha_params: Dict[str, float]) -> float:
    P_buf = E_buf = alpha_buf = result_buf = None
    try:
        alpha_values = np.array([alpha_params["alpha_O"], alpha_params["alpha_P"], alpha_params["alpha_E"]], dtype=np.float32)
        P_buf = create_buffer(context, np.float32(0).nbytes, np.array([P], dtype=np.float32))
        E_buf = create_buffer(context, np.float32(0).nbytes, np.array([E], dtype=np.float32))
        alpha_buf = create_buffer(context, alpha_values.nbytes, alpha_values)
        result_buf = create_buffer(context, np.float32(0).nbytes)
        operational_efficiency_program.calculate_operational_efficiency(queue, (1,), None, P_buf, E_buf, alpha_buf, result_buf)
        result = np.empty(1, dtype=np.float32)
        cl.enqueue_copy(queue, result, result_buf).wait()
        efficiency = result[0]
        logging.debug(f"calculate_operational_efficiency: Calculated operational efficiency: {efficiency}")
        logging.debug(f"P: {P}, E: {E}, Alpha Params: {alpha_params}, Result: {result}")
        return efficiency
    except cl.MemoryError as e:
        logging.error(f"calculate_operational_efficiency: MemoryError calculating operational efficiency: {e}")
        logging.debug(f"MemoryError details: {e}", exc_info=True)
        raise
    except Exception as e:
        logging.error(f"calculate_operational_efficiency: Error calculating operational efficiency: {e}")
        logging.debug(f"Error details: {e}", exc_info=True)
        raise
    finally:
        if P_buf: P_buf.release()
        if E_buf: E_buf.release()
        if alpha_buf: alpha_buf.release()
        if result_buf: result_buf.release()

# Function to calculate error management effectiveness (essentially the ability of the system to detect and correct errors)
def calculate_error_management(error_detection_rate: float, correction_capability: float, alpha_params: Dict[str, float]) -> float:
    error_detection_rate_buf = correction_capability_buf = alpha_buf = result_buf = None
    try:
        alpha_values = np.array([alpha_params["alpha_Em"], alpha_params["alpha_Error_Detection"], alpha_params["alpha_Correction"]], dtype=np.float32)
        error_detection_rate_buf = create_buffer(context, np.float32(0).nbytes, np.array([error_detection_rate], dtype=np.float32))
        correction_capability_buf = create_buffer(context, np.float32(0).nbytes, np.array([correction_capability], dtype=np.float32))
        alpha_buf = create_buffer(context, alpha_values.nbytes, alpha_values)
        result_buf = create_buffer(context, np.float32(0).nbytes)
        error_management_program.calculate_error_management(queue, (1,), None, error_detection_rate_buf, correction_capability_buf, alpha_buf, result_buf)
        result = np.empty(1, dtype=np.float32)
        cl.enqueue_copy(queue, result, result_buf).wait()
        error_management_value = result[0]
        logging.debug(f"calculate_error_management: Calculated error management effectiveness: {error_management_value}")
        logging.debug(f"Error Detection Rate: {error_detection_rate}, Correction Capability: {correction_capability}, Alpha Params: {alpha_params}, Result: {result}")
        return error_management_value
    except cl.MemoryError as e:
        logging.error(f"calculate_error_management: MemoryError calculating error management effectiveness: {e}")
        logging.debug(f"MemoryError details: {e}", exc_info=True)
        raise
    except Exception as e:
        logging.error(f"calculate_error_management: Error calculating error management effectiveness: {e}")
        logging.debug(f"Error details: {e}", exc_info=True)
        raise
    finally:
        if error_detection_rate_buf: error_detection_rate_buf.release()
        if correction_capability_buf: correction_capability_buf.release()
        if alpha_buf: alpha_buf.release()
        if result_buf: result_buf.release()

# Function to calculate adaptability metric (essentially the ability of the system to adapt to changes in its environment)
def calculate_adaptability(adaptation_rate: float, alpha_params: Dict[str, float]) -> float:
    adaptation_rate_buf = alpha_buf = result_buf = None
    try:
        alpha_values = np.array([alpha_params["alpha_A"], alpha_params["alpha_Adaptation_Rate"]], dtype=np.float32)
        adaptation_rate_buf = create_buffer(context, np.float32(0).nbytes, np.array([adaptation_rate], dtype=np.float32))
        alpha_buf = create_buffer(context, alpha_values.nbytes, alpha_values)
        result_buf = create_buffer(context, np.float32(0).nbytes)
        adaptability_program.calculate_adaptability(queue, (1,), None, adaptation_rate_buf, alpha_buf, result_buf)
        result = np.empty(1, dtype=np.float32)
        cl.enqueue_copy(queue, result, result_buf).wait()
        adaptability_value = result[0]
        logging.debug(f"calculate_adaptability: Calculated adaptability: {adaptability_value}")
        logging.debug(f"Adaptation Rate: {adaptation_rate}, Alpha Params: {alpha_params}, Result: {result}")
        return adaptability_value
    except cl.MemoryError as e:
        logging.error(f"calculate_adaptability: MemoryError calculating adaptability: {e}")
        logging.debug(f"MemoryError details: {e}", exc_info=True)
        raise
    except Exception as e:
        logging.error(f"calculate_adaptability: Error calculating adaptability: {e}")
        logging.debug(f"Error details: {e}", exc_info=True)
        raise
    finally:
        if adaptation_rate_buf: adaptation_rate_buf.release()
        if alpha_buf: alpha_buf.release()
        if result_buf: result_buf.release()

# Function to calculate volume metric (essentially the spatial processing resolution of the system)
def calculate_volume(spatial_scale: float, alpha_params: Dict[str, float]) -> float:
    spatial_scale_buf = alpha_buf = result_buf = None
    try:
        logging.debug(f"calculate_volume: Starting calculation with spatial_scale={spatial_scale}, alpha_params={alpha_params}")
        alpha_values = np.array([alpha_params["alpha_Volume"], alpha_params["alpha_Spatial_Scale"]], dtype=np.float32)
        spatial_scale_buf = create_buffer(context, np.float32(0).nbytes, np.array([spatial_scale], dtype=np.float32))
        alpha_buf = create_buffer(context, alpha_values.nbytes, alpha_values)
        result_buf = create_buffer(context, np.float32(0).nbytes)
        volume_program.calculate_volume(queue, (1,), None, spatial_scale_buf, alpha_buf, result_buf)
        result = np.empty(1, dtype=np.float32)
        cl.enqueue_copy(queue, result, result_buf).wait()
        volume_value = result[0]
        logging.debug(f"calculate_volume: Calculated volume: {volume_value}")
        return volume_value
    except cl.MemoryError as e:
        logging.error(f"calculate_volume: MemoryError calculating volume: {e}")
        with open('error.log', 'a') as error_log:
            error_log.write(f"calculate_volume: MemoryError calculating volume: {e}\n")
        raise
    except Exception as e:
        logging.error(f"calculate_volume: Error calculating volume: {e}")
        with open('error.log', 'a') as error_log:
            error_log.write(f"calculate_volume: Error calculating volume: {e}\n")
        raise
    finally:
        if spatial_scale_buf: spatial_scale_buf.release()
        if alpha_buf: alpha_buf.release()
        if result_buf: result_buf.release()

# Function to calculate time factor metric (essentially the temporal processing resolution of the system)
def calculate_time(temporal_scale: float, alpha_params: Dict[str, float]) -> float:
    temporal_scale_buf = alpha_buf = result_buf = None
    try:
        logging.debug(f"calculate_time: Starting calculation with temporal_scale={temporal_scale}, alpha_params={alpha_params}")
        alpha_values = np.array([alpha_params["alpha_t"], alpha_params["alpha_Temporal_Scale"]], dtype=np.float32)
        temporal_scale_buf = create_buffer(context, np.float32(0).nbytes, np.array([temporal_scale], dtype=np.float32))
        alpha_buf = create_buffer(context, alpha_values.nbytes, alpha_values)
        result_buf = create_buffer(context, np.float32(0).nbytes)
        time_program.calculate_time(queue, (1,), None, temporal_scale_buf, alpha_buf, result_buf)
        result = np.empty(1, dtype=np.float32)
        cl.enqueue_copy(queue, result, result_buf).wait()
        time_value = result[0]
        logging.debug(f"calculate_time: Calculated time: {time_value}")
        return time_value
    except cl.MemoryError as e:
        logging.error(f"calculate_time: MemoryError calculating time: {e}")
        with open('error.log', 'a') as error_log:
            error_log.write(f"calculate_time: MemoryError calculating time: {e}\n")
        raise
    except Exception as e:
        logging.error(f"calculate_time: Error calculating time: {e}")
        with open('error.log', 'a') as error_log:
            error_log.write(f"calculate_time: Error calculating time: {e}\n")
        raise
    finally:
        if temporal_scale_buf: temporal_scale_buf.release()
        if alpha_buf: alpha_buf.release()
        if result_buf: result_buf.release()
        logging.debug("calculate_time: Released all buffers")

# Function to process a combination of parameters and calculate the intelligence potential metric
def process_combination(combination: Tuple[np.ndarray, float, float, float, float, float, float, float, float, float, float]) -> pd.DataFrame:
    probabilities, H_X, H_Y, H_XY, P, E, error_detection_rate, correction_capability, adaptation_rate, spatial_scale, temporal_scale = combination
    logging.debug(f"process_combination: Processing combination {combination}")

    try:
        H_X_value = calculate_entropy(probabilities, alpha_parameters)
        I_XY_value = calculate_mutual_information(H_X, H_Y, H_XY, alpha_parameters)
        O_value = calculate_operational_efficiency(P, E, alpha_parameters)
        Em_value = calculate_error_management(error_detection_rate, correction_capability, alpha_parameters)
        A_value = calculate_adaptability(adaptation_rate, alpha_parameters)
        Volume_value = calculate_volume(spatial_scale, alpha_parameters)
        Time_value = calculate_time(temporal_scale, alpha_parameters)

        I = alpha_parameters["k"] * (H_X_value * I_XY_value * O_value * Em_value * A_value) / (Volume_value * Time_value)

        logging.debug(f"process_combination: Calculated values - H_X_value: {H_X_value}, I_XY_value: {I_XY_value}, O_value: {O_value}, Em_value: {Em_value}, A_value: {A_value}, Volume_value: {Volume_value}, Time_value: {Time_value}, Intelligence: {I}")

        return pd.DataFrame({
            'Probabilities': [probabilities], 'H_X': [H_X], 'H_Y': [H_Y], 'H_XY': [H_XY],
            'P': [P], 'E': [E], 'Error Detection Rate': [error_detection_rate],
            'Correction Capability': [correction_capability], 'Adaptation Rate': [adaptation_rate],
            'Spatial Scale': [spatial_scale], 'Temporal Scale': [temporal_scale], 'Intelligence': [I]
        })
    except Exception as e:
        logging.error(f"Error processing combination {combination}: {e}", exc_info=True)
        with open('error.log', 'a') as error_log:
            error_log.write(f"Error processing combination {combination}: {e}\n")
        raise

# Class to visualize the results
class DataVisualizer:
    def __init__(self, data: Optional[Union[pd.DataFrame, str]] = None) -> None:
        """
        Initialize the DataVisualizer with optional data.

        Args:
            data (Optional[Union[pd.DataFrame, str]]): The data to visualize, either as a pandas DataFrame or a CSV file path.
        """
        self.data: Optional[pd.DataFrame] = self._load_data(data) if data is not None else None
        if self.data is not None:
            logging.debug(f"DataVisualizer initialized with data of shape {self.data.shape} and columns {self.data.columns.tolist()}")
    
    def _load_data(self, data: Union[pd.DataFrame, str]) -> pd.DataFrame:
        """
        Load data from a pandas DataFrame or a CSV file path.

        Args:
            data (Union[pd.DataFrame, str]): The data to load.

        Returns:
            pd.DataFrame: The loaded data as a pandas DataFrame.

        Raises:
            ValueError: If the provided data is neither a DataFrame nor a valid CSV file path.
        """
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, str) and data.endswith('.csv'):
            try:
                return pd.read_csv(data)
            except Exception as e:
                logging.error(f"Failed to load data from {data}: {e}", exc_info=True)
                raise
        else:
            error_message = "The provided data must be a pandas DataFrame or a CSV file path."
            logging.error(error_message)
            raise ValueError(error_message)
    
    def _validate_columns(self, columns: List[str], data: Optional[pd.DataFrame] = None) -> None:
        """
        Validate that the specified columns exist in the data.

        Args:
            columns (List[str]): The columns to validate.
            data (Optional[pd.DataFrame], optional): The data to validate against. Defaults to None.

        Raises:
            ValueError: If no data is provided or if any columns are missing.
        """
        data = self._load_data(data) if data is not None else self.data
        if data is None:
            raise ValueError("No data provided for column validation.")
        missing_columns = [col for col in columns if col not in data.columns]
        if missing_columns:
            error_message = f"The following columns are missing in the dataset: {missing_columns}"
            logging.warning(error_message)
            raise ValueError(error_message)
    
    def _generate_filename(self, base_name: str) -> str:
        """
        Generate a filename with a timestamp.

        Args:
            base_name (str): The base name for the file.

        Returns:
            str: The generated filename with a timestamp.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base_name}_{timestamp}.png"
    
    def _save_and_show_plot(self, plot_type: str, filename: str, plot_object: Any = None, plot_library: str = 'matplotlib', **kwargs) -> None:
        """
        Save and display the plot using the specified plotting library.

        Args:
            plot_type (str): The type of plot being saved and displayed.
            filename (str): The name of the file to save the plot as.
            plot_object (Any, optional): The plot object to save and display. Defaults to None.
            plot_library (str, optional): The plotting library used to create the plot. Can be 'matplotlib', 'seaborn', 'plotly', 'joypy', or 'bokeh'. Defaults to 'matplotlib'.
            **kwargs: Additional keyword arguments to pass to the plot saving and displaying functions.
        """
        directory: str = os.path.join('/home/lloyd/UniversalIntelligencePotential', plot_type)
        os.makedirs(directory, exist_ok=True)
        filepath: str = os.path.join(directory, filename)
        
        plot_libraries = ['matplotlib', 'seaborn', 'plotly', 'joypy', 'bokeh']
        if plot_library not in plot_libraries:
            raise ValueError(f"Unsupported plotting library: {plot_library}")
        
        try:
            if plot_library == 'matplotlib':
                if plot_object is None:
                    plot_object = plt
                plot_object.gcf().set_size_inches(20, 15)  # Set maximum size for clarity
                plot_object.savefig(filepath, bbox_inches='tight', pad_inches=0.1, **kwargs)
                plot_object.show()
                plt.pause(0.001)  # Briefly show the plot
                plot_object.close()
            elif plot_library == 'seaborn':
                if plot_object is None:
                    raise ValueError("plot_object must be provided when using seaborn.")
                plot_object.figure.set_size_inches(20, 15)  # Set maximum size for clarity
                plot_object.figure.savefig(filepath, bbox_inches='tight', pad_inches=0.1, **kwargs)
                plot_object.figure.show()
                plt.pause(0.001)  # Briefly show the plot
                plt.close(plot_object.figure)
            elif plot_library == 'plotly':
                if plot_object is None:
                    raise ValueError("plot_object must be provided when using plotly.")
                plot_object.update_layout(width=1920, height=1080)  # Set maximum size for clarity
                plot_object.write_image(filepath, **kwargs)
                plot_object.show()
            elif plot_library == 'joypy':
                if plot_object is None:
                    raise ValueError("plot_object must be provided when using joypy.")
                fig, axes = plot_object  # Unpack the tuple returned by joypy.joyplot
                fig.set_size_inches(20, 15)  # Set maximum size for clarity
                fig.savefig(filepath, bbox_inches='tight', pad_inches=0.1, **kwargs)
                plt.show()
                plt.pause(0.001)  # Briefly show the plot
                plt.close(fig)
            elif plot_library == 'bokeh':
                if plot_object is None:
                    raise ValueError("plot_object must be provided when using bokeh.")
                plot_object.plot_width = 1920  # Set maximum size for clarity
                plot_object.plot_height = 1080
                save(plot_object, filename=filepath, resources=INLINE, **kwargs)
                show(plot_object)
            plt.close('all')  # Close all figures to manage memory
            logging.info(f"Plot saved as {filepath}")
        except Exception as e:
            logging.error(f"Failed to save and display plot using {plot_library}: {e}", exc_info=True)
            plt.close('all')  # Close all figures to manage memory
            raise RuntimeError(f"Failed to save and display plot using {plot_library}.") from e
        
    def plot_scatter_matrix(self, data: Optional[pd.DataFrame] = None, alpha: float = 0.8, figsize: Tuple[int, int] = (20, 15), diagonal: str = 'kde') -> None:
        """
        Plot a scatter matrix for all numeric columns in the data.

        Args:
            data (pd.DataFrame, optional): The data to plot. If not provided, the stored data will be used.
            alpha (float, optional): The transparency of the scatter points. Defaults to 0.8.
            figsize (tuple, optional): The figure size (width, height) in inches. Defaults to (20, 15).
            diagonal (str, optional): The type of plot to show on the diagonal. Can be 'hist' or 'kde'. Defaults to 'kde'.
        """
        logging.debug("Starting to plot scatter matrix.")
        
        # Load data if provided, otherwise use stored data
        data = self._load_data(data) if data is not None else self.data
        if data is None:
            raise ValueError("No data provided for plotting.")
        
        # Select numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            error_message = "No numeric columns available for scatter matrix."
            logging.warning(error_message)
            return
        
        try:
            # Calculate height for each subplot
            height_per_subplot = figsize[1] / len(numeric_data.columns)
            
            # Create scatter matrix plot
            scatter_matrix = sns.pairplot(
                numeric_data,
                diag_kind=diagonal,
                kind='scatter',
                height=height_per_subplot,
                aspect=figsize[0] / figsize[1],
                corner=True,
                plot_kws={'alpha': alpha, 'edgecolor': 'black', 'linewidth': 0.5},
                diag_kws={'fill': True}
            )
            
            # Customize the plot
            scatter_matrix.fig.suptitle('Scatter Matrix of Parameters and Intelligence', fontsize=16)
            scatter_matrix.fig.subplots_adjust(top=0.95)
            
            # Save and show the plot
            self._save_and_show_plot('scatter_matrix', self._generate_filename('scatter_matrix'), plot_object=scatter_matrix, plot_library='seaborn')
        
        except Exception as e:
            logging.error(f"Failed to plot scatter matrix: {e}", exc_info=True)
            plt.close('all')  # Ensure all figures are closed to manage memory
            raise RuntimeError(f"Failed to plot scatter matrix.") from e
        
    def plot_ridge_plot(self, data: Optional[pd.DataFrame] = None, by: str = 'Intelligence', figsize: Tuple[int, int] = (12, 8), colormap: str = 'viridis', alpha: float = 0.8) -> None:
        """
        Plot a ridge plot for all numeric columns, grouped by the specified column.

        Args:
            data (pd.DataFrame, optional): The data to plot. If not provided, the stored data will be used.
            by (str, optional): The column to group the data by. Defaults to 'Intelligence'.
            figsize (tuple, optional): The figure size (width, height) in inches. Defaults to (12, 8).
            colormap (str, optional): The colormap to use for the plot. Defaults to 'viridis'.
            alpha (float, optional): The transparency of the plot. Defaults to 0.8.
        """
        logging.debug(f"Starting to plot ridge plot grouped by {by}.")
        
        # Load data if not provided
        data = self._load_data(data) if data is not None else self.data
        if data is None:
            raise ValueError("No data provided for plotting.")
        
        # Select numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            error_message = "No numeric columns available for ridge plot."
            logging.warning(error_message)
            return
        
        # Validate the grouping column
        self._validate_columns([by], data=data)
        
        try:
            # Ensure colormap is a callable
            if isinstance(colormap, str):
                colormap = plt.get_cmap(colormap)
            
            # Create the ridge plot
            fig, axes = joypy.joyplot(data, by=by, figsize=figsize, colormap=colormap, alpha=alpha, linewidth=1, legend=True)
            
            # Customize the plot
            plt.title(f'Ridge Plot of Parameters Grouped by {by}', fontsize=16)
            plt.xlabel('Value', fontsize=14)
            plt.ylabel('Parameter', fontsize=14)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.grid(alpha=0.3)
            
            # Save and show the plot
            self._save_and_show_plot('ridge_plots', self._generate_filename(f'ridge_plot_by_{by}'), plot_object=(fig, axes), plot_library='joypy')
        
        except TypeError as e:
            if "'str' object is not callable" in str(e):
                logging.error(f"Colormap error: {e}. Ensure colormap is a valid colormap object.")
                raise ValueError("Invalid colormap provided. Ensure colormap is a valid colormap object.") from e
            else:
                logging.error(f"Failed to plot ridge plot: {e}", exc_info=True)
                raise RuntimeError(f"Failed to plot ridge plot.") from e
        except Exception as e:
            logging.error(f"Failed to plot ridge plot: {e}", exc_info=True)
            raise RuntimeError(f"Failed to plot ridge plot.") from e
        
    def plot_3d_scatter(self, data: Optional[pd.DataFrame] = None, x: Optional[str] = None, y: Optional[str] = None, z: Optional[str] = None, color: str = 'blue', marker: str = 'o', figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Plot a 3D scatter plot for the specified columns or all numeric column combinations.

        Args:
            data (pd.DataFrame, optional): The data to plot. If not provided, the stored data will be used.
            x (str, optional): The column to use for the x-axis. If not provided, all numeric columns will be used.
            y (str, optional): The column to use for the y-axis. If not provided, all numeric columns will be used.
            z (str, optional): The column to use for the z-axis. If not provided, 'Intelligence' will be used.
            color (str, optional): The color of the scatter points. Defaults to 'blue'.
            marker (str, optional): The marker style for the scatter points. Defaults to 'o' (circle).
            figsize (tuple, optional): The figure size (width, height) in inches. Defaults to (10, 8).
        """
        logging.debug(f"Starting to plot 3D scatter plot with x={x}, y={y}, and z={z}.")
        
        # Load data if not provided
        data = self._load_data(data) if data is not None else self.data
        if data is None:
            raise ValueError("No data provided for plotting.")
        
        # Select numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            logging.warning("No numeric columns available for 3D scatter plot.")
            return
        
        # If no specific columns are provided, plot all combinations
        if x is None and y is None and z is None:
            for x_col, y_col, z_col in itertools.permutations(numeric_data.columns, 3):
                try:
                    fig = plt.figure(figsize=figsize)
                    ax = fig.add_subplot(111, projection='3d')
                    ax.scatter(data[x_col], data[y_col], data[z_col], c=color, marker=marker, alpha=0.8, edgecolors='black', linewidths=0.5)
                    ax.set_xlabel(x_col, fontsize=14)
                    ax.set_ylabel(y_col, fontsize=14)
                    ax.set_zlabel(z_col, fontsize=14)
                    plt.title(f'3D Scatter Plot of {x_col}, {y_col} and {z_col}', fontsize=16)
                    self._save_and_show_plot('3d_scatter_plots', self._generate_filename(f'{x_col}_vs_{y_col}_vs_{z_col}_3d_scatter'), plot_object=fig, plot_library='matplotlib')
                except Exception as e:
                    logging.error(f"Failed to plot 3D scatter for {x_col} vs {y_col} vs {z_col}: {e}", exc_info=True)
        else:
            # Use specified columns or default to first available numeric columns
            x = x or numeric_data.columns[0]
            y = y or numeric_data.columns[1]
            z = z or 'Intelligence'
            
            # Validate the specified columns
            self._validate_columns([x, y, z], data=data)
            
            try:
                fig = plt.figure(figsize=figsize)
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(data[x], data[y], data[z], c=color, marker=marker, alpha=0.8, edgecolors='black', linewidths=0.5)
                ax.set_xlabel(x, fontsize=14)
                ax.set_ylabel(y, fontsize=14)
                ax.set_zlabel(z, fontsize=14)
                plt.title(f'3D Scatter Plot of {x}, {y} and {z}', fontsize=16)
                self._save_and_show_plot('3d_scatter_plots', self._generate_filename(f'{x}_vs_{y}_vs_{z}_3d_scatter'), plot_object=fig, plot_library='matplotlib')
            except Exception as e:
                logging.error(f"Failed to plot 3D scatter: {e}", exc_info=True)
                raise RuntimeError("Failed to plot 3D scatter.") from e
            
    def plot_heatmap(self, data: Optional[pd.DataFrame] = None, figsize: Tuple[int, int] = (14, 10), cmap: str = 'coolwarm', annot: bool = True, linewidths: float = 0.5, fmt: str = '.2f', square: bool = True) -> None:
        """
        Plot a heatmap of the correlation matrix for the numeric columns in the dataset.

        Args:
            data (pd.DataFrame, optional): The data to plot. If not provided, the stored data will be used.
            figsize (tuple, optional): The figure size (width, height) in inches. Defaults to (14, 10).
            cmap (str, optional): The colormap to use for the heatmap. Defaults to 'coolwarm'.
            annot (bool, optional): Whether to annotate the heatmap with the correlation coefficients. Defaults to True.
            linewidths (float, optional): Width of the lines that will divide each cell. Defaults to 0.5.
            fmt (str, optional): String formatting code to use when adding annotations. Defaults to '.2f'.
            square (bool, optional): If True, set the Axes aspect to "equal" so each cell will be square-shaped. Defaults to True.
        """
        logging.debug("Starting to plot heatmap.")
        
        # Load data if provided, otherwise use stored data
        data = self._load_data(data) if data is not None else self.data
        if data is None:
            raise ValueError("No data provided for plotting.")
        
        try:
            # Select numeric columns
            numeric_data = data.select_dtypes(include=[np.number])
            if numeric_data.empty:
                logging.warning("No numeric columns available for correlation matrix.")
                return
            
            # Compute the correlation matrix
            correlation_matrix = numeric_data.corr()
            
            # Plot the heatmap
            plt.figure(figsize=figsize)
            heatmap = sns.heatmap(correlation_matrix, cmap=cmap, annot=annot, linewidths=linewidths, fmt=fmt, square=square, cbar_kws={'shrink': 0.5})
            heatmap.set_title('Correlation Matrix', fontsize=16)
            heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, fontsize=12)
            heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, fontsize=12)
            
            # Save and show the plot
            self._save_and_show_plot('heatmap', self._generate_filename('correlation_matrix'), plot_object=heatmap, plot_library='seaborn')
        
        except Exception as e:
            logging.error(f"Failed to plot heatmap: {e}", exc_info=True)
            raise RuntimeError("Failed to plot heatmap.") from e
    
    def plot_pairplot(self, data: Optional[pd.DataFrame] = None, hue: str = 'Intelligence', palette: str = 'viridis', height: float = 2.5, aspect: float = 1, corner: bool = False) -> None:
        """
        Plot a pairplot for the numeric columns in the dataset.

        Args:
            data (pd.DataFrame, optional): The data to plot. If not provided, the stored data will be used.
            hue (str, optional): Variable in `data` to map plot aspects to different colors. Defaults to 'Intelligence'.
            palette (str, optional): Colors to use for the different levels of the `hue` variable. Defaults to 'viridis'.
            height (float, optional): Height (in inches) of each facet. Defaults to 2.5.
            aspect (float, optional): Aspect ratio of each facet, so that aspect * height gives the width of each facet in inches. Defaults to 1.
            corner (bool, optional): If True, don't add axes to the upper (off-diagonal) triangle of the grid, making this a "corner" plot. Defaults to False.
        """
        logging.debug(f"Starting to plot pairplot with hue '{hue}'.")
        
        # Load data if provided, otherwise use stored data
        data = self._load_data(data) if data is not None else self.data
        if data is None:
            raise ValueError("No data provided for plotting.")
        
        try:
            # Select numeric columns
            numeric_data = data.select_dtypes(include=[np.number])
            if numeric_data.empty:
                logging.warning("No numeric columns available for pairplot.")
                return
            
            # Validate hue column
            if hue not in data.columns:
                logging.warning(f"Hue column '{hue}' not found in data. Defaulting to 'Intelligence'.")
                hue = 'Intelligence'
            
            # Plot the pairplot
            g = sns.pairplot(data, hue=hue, palette=palette, height=height, aspect=aspect, corner=corner, diag_kind='kde', plot_kws={'alpha': 0.7})
            g.fig.suptitle('Pairplot of Parameters and Intelligence', fontsize=16)
            g.fig.subplots_adjust(top=0.9)
            
            # Save and show the plot
            self._save_and_show_plot('pairplot', self._generate_filename('pairplot'), plot_object=g, plot_library='seaborn')
        
        except Exception as e:
            logging.error(f"Failed to plot pairplot: {e}", exc_info=True)
            raise RuntimeError("Failed to plot pairplot.") from e    
    
    def plot_jointplot(self, data: Optional[pd.DataFrame] = None, x: Optional[str] = None, y: str = 'Intelligence', kind: str = 'hex', cmap: str = 'Blues', height: float = 8) -> None:
        """
        Plot a jointplot for the specified x and y columns in the dataset.

        Args:
            data (pd.DataFrame, optional): The data to plot. If not provided, the stored data will be used.
            x (str, optional): The column to plot on the x-axis. If not provided, all numeric columns will be used.
            y (str, optional): The column to plot on the y-axis. Defaults to 'Intelligence'.
            kind (str, optional): The kind of plot to draw. Defaults to 'hex'.
            cmap (str, optional): The colormap to use for the plot. Defaults to 'Blues'.
            height (float, optional): The height (in inches) of the plot. Defaults to 8.
        """
        logging.debug(f"Starting to plot jointplot with x={x} and y={y}.")
        
        # Load data if provided, otherwise use stored data
        data = self._load_data(data) if data is not None else self.data
        if data is None:
            raise ValueError("No data provided for plotting.")
        
        # Select numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            logging.warning("No numeric columns available for jointplot.")
            return
        
        # Plot jointplot for each numeric column if x is not specified
        if x is None:
            for x_col in numeric_data.columns:
                if x_col != y:
                    try:
                        g = sns.jointplot(data=data, x=x_col, y=y, kind=kind, cmap=cmap, height=height, marginal_ticks=True, joint_kws=dict(alpha=0.7))
                        g.set_axis_labels(x_col, y, fontsize=14)
                        g.fig.suptitle(f'Jointplot of {x_col} vs {y}', fontsize=16)
                        g.fig.subplots_adjust(top=0.9)
                        self._save_and_show_plot('jointplot', self._generate_filename(f'jointplot_{x_col}_vs_{y}'), plot_object=g, plot_library='seaborn')
                    except Exception as e:
                        logging.error(f"Failed to plot jointplot for {x_col} vs {y}: {e}", exc_info=True)
        else:
            self._validate_columns([x, y], data=data)
            try:
                g = sns.jointplot(data=data, x=x, y=y, kind=kind, cmap=cmap, height=height, marginal_ticks=True, joint_kws=dict(alpha=0.7))
                g.set_axis_labels(x, y, fontsize=14)
                g.fig.suptitle(f'Jointplot of {x} vs {y}', fontsize=16)
                g.fig.subplots_adjust(top=0.9)
                self._save_and_show_plot('jointplot', self._generate_filename(f'jointplot_{x}_vs_{y}'), plot_object=g, plot_library='seaborn')
            except Exception as e:
                logging.error(f"Failed to plot jointplot: {e}", exc_info=True)
                raise RuntimeError("Failed to plot jointplot.") from e
    
    def plot_histograms(self, data: Optional[pd.DataFrame] = None, bins: int = 10, kde: bool = True, color: Optional[str] = None, figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot histograms for all numeric columns in the dataset.

        Args:
            data (pd.DataFrame, optional): The data to plot. If not provided, the stored data will be used.
            bins (int, optional): Number of bins for the histogram. Defaults to 10.
            kde (bool, optional): Whether to plot a kernel density estimate. Defaults to True.
            color (str, optional): Color for the histogram bars. Defaults to None.
            figsize (tuple, optional): The figure size (width, height) in inches. Defaults to (10, 6).
        """
        logging.debug("Starting to plot histograms.")
        
        # Load data if provided, otherwise use stored data
        data = self._load_data(data) if data is not None else self.data
        if data is None:
            raise ValueError("No data provided for plotting.")
        
        # Select numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            logging.warning("No numeric columns available for histograms.")
            return
        
        try:
            for column in numeric_data.columns:
                logging.debug(f"Plotting histogram for column: {column}")
                plt.figure(figsize=figsize)
                histogram = sns.histplot(
                    numeric_data[column], 
                    bins=bins, 
                    kde=kde, 
                    color=color, 
                    stat='density', 
                    alpha=0.7, 
                    edgecolor='black', 
                    linewidth=1
                )
                plt.title(f'Distribution of {column}', fontsize=16)
                plt.xlabel(column, fontsize=14)
                plt.ylabel('Density', fontsize=14)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                plt.grid(alpha=0.3)
                self._save_and_show_plot(
                    'histograms', 
                    self._generate_filename(f'{column}_distribution'), 
                    plot_object=histogram, 
                    plot_library='seaborn'
                )
        except Exception as e:
            logging.error(f"Failed to plot histograms: {e}", exc_info=True)
            raise RuntimeError("Failed to plot histograms.") from e  
    
    def plot_boxplots(self, data: Optional[pd.DataFrame] = None, x: str = 'Intelligence', figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot boxplots for all numeric columns against a specified column.

        Args:
            data (pd.DataFrame, optional): The data to plot. If not provided, the stored data will be used.
            x (str): The column to use for the x-axis. Defaults to 'Intelligence'.
            figsize (tuple): The figure size (width, height) in inches. Defaults to (12, 8).
        """
        logging.debug(f"Starting to plot boxplots with x={x}.")
        
        # Load data if provided, otherwise use stored data
        data = self._load_data(data) if data is not None else self.data
        if data is None:
            raise ValueError("No data provided for plotting.")
        
        # Select numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            logging.warning("No numeric columns available for boxplots.")
            return
        
        # Validate the x column
        self._validate_columns([x], data=data)
        
        try:
            for column in numeric_data.columns:
                if column != x:
                    logging.debug(f"Plotting boxplot for column: {column}")
                    plt.figure(figsize=figsize)
                    boxplot = sns.boxplot(
                        x=x, 
                        y=column, 
                        data=data, 
                        palette='viridis', 
                        linewidth=1.5, 
                        fliersize=3, 
                        whis=1.5
                    )
                    plt.title(f'{x} vs {column}', fontsize=16)
                    plt.xlabel(x, fontsize=14)
                    plt.ylabel(column, fontsize=14)
                    plt.xticks(fontsize=12)
                    plt.yticks(fontsize=12)
                    plt.grid(alpha=0.3)
                    self._save_and_show_plot(
                        'boxplots', 
                        self._generate_filename(f'{x}_vs_{column}'), 
                        plot_object=boxplot, 
                        plot_library='seaborn'
                    )
        except Exception as e:
            logging.error(f"Failed to plot boxplots: {e}", exc_info=True)
            raise RuntimeError("Failed to plot boxplots.") from e   
        
    def plot_violinplot(self, data: Optional[pd.DataFrame] = None, x: Optional[str] = None, y: str = 'Intelligence', scale: str = 'width', inner: str = 'quartile', figsize: Tuple[int, int] = (20, 15)) -> None:
        """
        Plot a violinplot for the specified columns or all numeric column combinations.

        Args:
            data (pd.DataFrame, optional): The data to plot. If not provided, the stored data will be used.
            x (str, optional): The column to use for the x-axis. If not provided, all numeric columns will be used.
            y (str, optional): The column to use for the y-axis. Defaults to 'Intelligence'.
            scale (str, optional): The method used to scale the width of each violin. Can be 'width' or 'area'. Defaults to 'width'.
            inner (str, optional): The representation of the datapoints in the violin interior. Can be 'quartile', 'box', 'point', 'stick', or None. Defaults to 'quartile'.
            figsize (tuple, optional): The figure size (width, height) in inches. Defaults to (20, 15).
        """
        logging.debug(f"Starting to plot violinplot with x={x} and y={y}.")
        data = self._load_data(data) if data is not None else self.data
        if data is None:
            raise ValueError("No data provided for plotting.")
        
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            error_message = "No numeric columns available for violinplot."
            logging.warning(error_message)
            return 
        
        def plot_violin(x_col: str) -> None:
            """Helper function to plot a violin plot for a given x column."""
            try:
                plt.figure(figsize=figsize)
                violinplot = sns.violinplot(
                    data=data, x=x_col, y=y, scale=scale, inner=inner, 
                    palette='viridis', linewidth=1.5, cut=0
                )
                plt.title(f'{x_col} vs {y}', fontsize=16)
                plt.xlabel(x_col, fontsize=14)
                plt.ylabel(y, fontsize=14)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                plt.grid(alpha=0.3)
                self._save_and_show_plot(
                    'violinplots', 
                    self._generate_filename(f'{x_col}_vs_{y}_violin'), 
                    plot_object=violinplot, 
                    plot_library='seaborn'
                )
            except Exception as e:
                logging.error(f"Failed to plot violinplot for {x_col} vs {y}: {e}", exc_info=True)
        
        if x is None:
            for x_col in numeric_data.columns:
                if x_col != y:
                    plot_violin(x_col)
        else:
            self._validate_columns([x, y], data=data)
            plot_violin(x)
    
    def visualize_results(self, data: Optional[pd.DataFrame] = None) -> None:
        """
        Visualize the results using various plotting methods.

        Args:
            data (pd.DataFrame, optional): The data to visualize. If not provided, the stored data will be used.
        """
        logging.debug("Starting to visualize results.")
        
        # Load data if provided, otherwise use stored data
        data = self._load_data(data) if data is not None else self.data
        if data is None:
            raise ValueError("No data provided for visualization.")
        
        try:
            # Ensure the data is a pandas DataFrame
            if not isinstance(data, pd.DataFrame):
                error_message = "The data attribute must be a pandas DataFrame."
                logging.error(error_message)
                raise ValueError(error_message)
            
            # Select numeric columns for visualization
            numeric_results = data.select_dtypes(include=[np.number])
            if numeric_results.empty:
                error_message = "No numeric columns available for visualization."
                logging.warning(error_message)
                return
            
            logging.debug(f"Numeric columns selected for visualization: {numeric_results.columns.tolist()}")

            # List of visualization methods to be applied
            visualization_methods = [
                self.plot_violinplot,
                self.plot_scatter_matrix,                
                self.plot_heatmap,
                self.plot_pairplot,
                self.plot_jointplot,
                self.plot_histograms,
                self.plot_boxplots,
                self.plot_3d_scatter,
                self.plot_ridge_plot,                
            ]

            # Apply each visualization method
            for method in visualization_methods:
                try:
                    method(data=data)
                    logging.info(f"{method.__name__} plotted successfully.")
                    plt.close('all')  # Close all figures to manage memory
                except Exception as e:
                    logging.error(f"Failed to plot {method.__name__}: {e}", exc_info=True)
                    raise

        except Exception as e:
            logging.error(f"Failed to visualize results: {e}", exc_info=True)
        finally:
            logging.info("Visualization process completed.")
            plt.close('all')  # Close all figures to manage memory

# Main function to execute the universal intelligence potential model
def main() -> None:
    """
    Main function to execute the universal intelligence potential model.
    This function loads or generates parameter ranges, processes combinations of parameters,
    and visualizes the results.
    """
    visualizer = DataVisualizer()
    try:
        logging.info("Main execution started.")
        
        # Load or generate parameter ranges
        parameter_file = 'advanced_parameter_ranges.npz'
        if os.path.exists(parameter_file):
            with np.load(parameter_file) as data:
                probabilities_set = data['probabilities_set']
                H_X_set = data['H_X_set']
                H_Y_set = data['H_Y_set']
                H_XY_set = data['H_XY_set']
                P_set = data['P_set']
                E_set = data['E_set']
                error_detection_rate_set = data['error_detection_rate_set']
                correction_capability_set = data['correction_capability_set']
                adaptation_rate_set = data['adaptation_rate_set']
                spatial_scale_set = data['spatial_scale_set']
                temporal_scale_set = data['temporal_scale_set']
            logging.debug("Loaded parameter ranges from file.")
        else:
            probabilities_set = [
                np.array([p1, p2, 1 - p1 - p2]) 
                for p1 in np.linspace(0.1, 0.9, 9) 
                for p2 in np.linspace(0.1, 0.9, 9) 
                if p1 + p2 <= 1
            ]
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

            # Save the refined parameter ranges to a file for future use
            np.savez(parameter_file, 
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
            logging.debug("Generated and saved new parameter ranges.")

        # Prepare a DataFrame with enhanced structure to store the results
        import pandas as pd
        results = pd.DataFrame(columns=[
            'Probabilities', 'H_X', 'H_Y', 'H_XY', 'P', 'E', 'Error Detection Rate',
            'Correction Capability', 'Adaptation Rate', 'Spatial Scale', 'Temporal Scale', 'Intelligence'
        ])
        logging.debug("Initialized results DataFrame.")

        # Generate all combinations of input parameters with improved efficiency
        all_combinations = list(itertools.product(
            probabilities_set, H_X_set, H_Y_set, H_XY_set, P_set, E_set,
            error_detection_rate_set, correction_capability_set, adaptation_rate_set,
            spatial_scale_set, temporal_scale_set
        ))
        logging.debug(f"Generated all combinations of input parameters. Total combinations: {len(all_combinations)}")

        # Initialize progress bar
        total_combinations = len(all_combinations)
        total_operations = total_combinations * 7  # 7 calculations per combination
        progress_bar = tqdm(total=total_operations, desc="Processing Combinations", unit="op", dynamic_ncols=True)
        start_time = time.time()

        # Process combinations in chunks to ensure scalability
        combinations_to_process = 10000 # Define the number of combinations to process from the total set
        chunk_size = int(combinations_to_process / 7)  # Define a chunk size for processing (taking into account that each combination takes 7 operations)
        for chunk_start in range(0, total_combinations, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_combinations)
            chunk_combinations = all_combinations[chunk_start:chunk_end]
            logging.debug(f"Processing chunk {chunk_start // chunk_size + 1} with combinations {chunk_start} to {chunk_end - 1}")

            for idx, combination in enumerate(chunk_combinations):
                result = process_combination(combination)
                # Only concatenate non-empty and non-all-NA results
                if not result.empty and not result.isna().all().all():
                    results = pd.concat([results, result], ignore_index=True)
                progress_bar.update(7)  # Update by 7 for each combination processed
                elapsed_time = time.time() - start_time
                progress_bar.set_postfix(ops_per_sec=f"{progress_bar.n / elapsed_time:.2f}")
                logging.debug(f"Processed combination {chunk_start + idx + 1}/{total_combinations}.")


            # Save intermediate results to ensure data persistence
            results.to_csv(f'results_chunk_{chunk_start // chunk_size + 1}.csv', index=False)
            logging.debug(f"Saved results for chunk {chunk_start // chunk_size + 1}.")

            # Pause the program for a period of time to ensure all processes for this loop have finished processing properly
            # and to ensure system resources have cleared properly.
            time.sleep(5)  # Pausing for 5 seconds. Adjust the duration as necessary based on system performance and requirements.

            # Visualise Intermediate Results and Save for a graphical progress record.
            try:
                visualizer.visualize_results(results)
            except Exception as e:
                logging.error(f"Failed to visualize results: {e}", exc_info=True)
            # Pause the program for a period of time to ensure all processes for this loop have finished processing properly
            # and to ensure system resources have cleared properly.
            time.sleep(5)  # Pausing for 5 seconds. Adjust the duration as necessary based on system performance and requirements.
        
        progress_bar.close()

        # Combine all chunk results into a final DataFrame
        final_results = pd.concat([pd.read_csv(f'results_chunk_{i + 1}.csv') for i in range((total_combinations + chunk_size - 1) // chunk_size)], ignore_index=True)
        final_results.to_csv('final_results.csv', index=False)
        logging.debug("Combined all chunk results into final results DataFrame.")

        # Output the results as a formatted table
        print(final_results.to_string(index=False))

        visualizer.visualize_results(results)
    except Exception as e:
        logging.error(f"Error in main execution: {e}", exc_info=True)
        with open('error.log', 'a') as error_log:
            error_log.write(f"Error in main execution: {e}\n")
        raise
    finally:
        logging.info("Main execution completed.")

if __name__ == "__main__":
    main()