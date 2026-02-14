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

# Configure logging to capture debug information for tracing computation values and include the current operation process with each value
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the parameters for the model with explicit type annotations
alpha_parameters: Dict[str, float] = {
    "k": 1.0, "alpha_H": 1.0, "alpha_I": 1.0, "alpha_O": 1.0, "alpha_Em": 1.0,
    "alpha_A": 1.0, "alpha_Volume": 1.0, "alpha_t": 1.0, "alpha_Pi": 1.0,
    "alpha_log": 1.0, "alpha_HX": 1.0, "alpha_HY": 1.0, "alpha_HXY": 1.0,
    "alpha_P": 1.0, "alpha_E": 1.0, "alpha_Error_Detection": 1.0,
    "alpha_Correction": 1.0, "alpha_Adaptation_Rate": 1.0, "alpha_Spatial_Scale": 1.0,
    "alpha_Temporal_Scale": 1.0
}

# Initialize progress bar for overall process
total_operations = 100  # Placeholder for total operations, adjust as needed
progress_bar = tqdm(total=total_operations, desc="Overall Progress", unit="operation")

# OpenCL setup and kernel management
class OpenCLManager:
    def __init__(self):
        self.context, self.queue = self.setup_opencl_environment()
        self.programs = {}
        self.load_all_programs()

    def setup_opencl_environment(self) -> Tuple[cl.Context, cl.CommandQueue]:
        try:
            platform = cl.get_platforms()[0]
            device = platform.get_devices()[0]
            context = cl.Context([device])
            queue = cl.CommandQueue(context)
            return context, queue
        except Exception as e:
            logging.error(f"Error setting up OpenCL environment: {e}")
            raise

    def load_kernel(self, filename: str) -> str:
        try:
            with open(filename, 'r') as file:
                return file.read()
        except Exception as e:
            logging.error(f"Error loading kernel from {filename}: {e}")
            raise

    def load_and_build_program(self, kernel_path: str, global_size: Tuple[int], local_size: Optional[Tuple[int]] = None) -> cl.Program:
        try:
            kernel_code = self.load_kernel(kernel_path)
            program = cl.Program(self.context, kernel_code).build()
            program.global_size = global_size
            program.local_size = local_size
            return program
        except Exception as e:
            logging.error(f"Failed to load and build program from {kernel_path}: {e}")
            raise

    def load_all_programs(self):
        kernel_directory = "/home/lloyd/UniversalIntelligencePotential/kernels/"
        kernel_files = ['entropy_kernel.cl', 'mutual_information_kernel.cl', 'operational_efficiency_kernel.cl',
                        'error_management_kernel.cl', 'adaptability_kernel.cl', 'volume_kernel.cl', 'time_kernel.cl']
        device = self.context.devices[0]
        max_work_group_size = device.max_work_group_size
        max_work_item_sizes = device.max_work_item_sizes
        global_size = (max_work_group_size * max_work_item_sizes[0],)
        local_size = (max_work_group_size,)

        for kernel_file in kernel_files:
            program_name = kernel_file.split('_')[0]
            self.programs[program_name] = self.load_and_build_program(kernel_directory + kernel_file, global_size, local_size)

# Parameter management and computation
class ParameterManager:
    def __init__(self, opencl_manager: OpenCLManager):
        self.opencl_manager = opencl_manager
        self.parameter_ranges = self.load_parameter_ranges()

    def load_parameter_ranges(self) -> Dict[str, np.ndarray]:
        if os.path.exists('parameter_ranges.npz'):
            with np.load('parameter_ranges.npz', allow_pickle=True) as data:
                return {key: data[key] for key in data}
        else:
            return self.create_and_save_parameter_ranges()

    def create_and_save_parameter_ranges(self) -> Dict[str, np.ndarray]:
        parameter_ranges = {
            'probabilities_set': self.generate_probabilities_set(),
            'H_X_set': np.linspace(0.2, 2.0, 5),
            'H_Y_set': np.linspace(0.2, 2.0, 5),
            'H_XY_set': np.linspace(0.5, 3.0, 5),
            'P_set': np.linspace(100.0, 2000.0, 5),
            'E_set': np.linspace(20.0, 100.0, 5),
            'error_detection_rate_set': np.linspace(0.5, 1.0, 3),
            'correction_capability_set': np.linspace(0.5, 1.0, 3),
            'adaptation_rate_set': np.linspace(0.3, 1.0, 4),
            'spatial_scale_set': np.linspace(0.5, 1.5, 3),
            'temporal_scale_set': np.linspace(0.5, 1.5, 3)
        }
        np.savez('parameter_ranges.npz', **parameter_ranges)
        return parameter_ranges

    def generate_probabilities_set(self) -> Dict[str, np.ndarray]:
        return {
            f'prob_{p1}_{p2}': np.array([p1, p2, 1 - p1 - p2])
            for p1 in np.linspace(0.1, 0.9, 9)
            for p2 in np.linspace(0.1, 0.9, 9)
            if p1 + p2 <= 1
        }

    def calculate_total_operations(self) -> int:
        combinations = self.generate_combinations()
        return len(combinations)

    def generate_combinations(self) -> List[Dict[str, Any]]:
        keys = list(self.parameter_ranges.keys())
        values = [self.parameter_ranges[key] for key in keys]
        return [dict(zip(keys, combination)) for combination in itertools.product(*values)]
    

CONTEXT, QUEUE = OpenCLManager.setup_opencl_environment()
# Dynamically determine the global and local work sizes based on device specifications
device = CONTEXT.devices[0]
max_work_group_size = device.max_work_group_size
max_work_item_sizes = device.max_work_item_sizes
GLOBAL_SIZE = (max_work_group_size * max_work_item_sizes[0],)
LOCAL_SIZE = (max_work_group_size,)
logging.debug(f"Determined global work size: {GLOBAL_SIZE}")
logging.debug(f"Determined local work size: {LOCAL_SIZE}")
kernel_directory: str = "/home/lloyd/UniversalIntelligencePotential/kernels/"

try:
    logging.debug("Loading and building OpenCL programs.")
    entropy_program = OpenCLManager.load_and_build_program(CONTEXT, kernel_directory + 'entropy_kernel.cl', GLOBAL_SIZE, LOCAL_SIZE)
    mutual_information_program = OpenCLManager.load_and_build_program(CONTEXT, kernel_directory + 'mutual_information_kernel.cl', GLOBAL_SIZE, LOCAL_SIZE)
    operational_efficiency_program = OpenCLManager.load_and_build_program(CONTEXT, kernel_directory + 'operational_efficiency_kernel.cl', GLOBAL_SIZE, LOCAL_SIZE)
    error_management_program = OpenCLManager.load_and_build_program(CONTEXT, kernel_directory + 'error_management_kernel.cl', GLOBAL_SIZE, LOCAL_SIZE)
    adaptability_program = OpenCLManager.load_and_build_program(CONTEXT, kernel_directory + 'adaptability_kernel.cl', GLOBAL_SIZE, LOCAL_SIZE)
    volume_program = OpenCLManager.load_and_build_program(CONTEXT, kernel_directory + 'volume_kernel.cl', GLOBAL_SIZE, LOCAL_SIZE)
    time_program = OpenCLManager.load_and_build_program(CONTEXT, kernel_directory + 'time_kernel.cl', GLOBAL_SIZE, LOCAL_SIZE)
    logging.debug("All OpenCL programs loaded and built successfully.")
except Exception as e:
    logging.error(f"Error in loading and building programs: {e}")
    raise

def calculate_entropy(probabilities: Union[np.ndarray, List[float]], alpha_params: Dict[str, float], context: cl.Context = CONTEXT, queue: cl.CommandQueue = QUEUE) -> float:
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
        
        # Use the stored global and local sizes
        global_size = entropy_program.global_size
        local_size = entropy_program.local_size
        
        entropy_program.calculate_entropy(queue, global_size, local_size, prob_buf, alpha_buf, result_buf, np.int32(len(probabilities)))
        
        result = np.empty_like(probabilities)
        cl.enqueue_copy(queue, result, result_buf).wait()
        entropy = np.sum(result)
        logging.debug(f"calculate_entropy: Calculated Shannon entropy: {entropy}")
        return entropy
    except Exception as e:
        logging.error(f"calculate_entropy: Error calculating Shannon entropy: {e}")
        raise

def calculate_mutual_information(H_X: Union[float, np.ndarray], H_Y: Union[float, np.ndarray], H_XY: Union[float, np.ndarray], alpha_params: Dict[str, float], context: cl.Context = CONTEXT, queue: cl.CommandQueue = QUEUE) -> float:
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
        
        # Use the stored global and local sizes
        global_size = mutual_information_program.global_size
        local_size = mutual_information_program.local_size
        
        mutual_information_program.calculate_mutual_information(queue, global_size, local_size, H_X_buf, H_Y_buf, H_XY_buf, alpha_buf, result_buf)
        
        result = np.empty_like(H_X)
        cl.enqueue_copy(queue, result, result_buf).wait()
        mutual_info = np.sum(result)
        logging.debug(f"calculate_mutual_information: Calculated mutual information: {mutual_info}")
        return mutual_info
    except Exception as e:
        logging.error(f"calculate_mutual_information: Error calculating mutual information: {e}")
        raise

def calculate_operational_efficiency(P: Union[float, np.ndarray], E: Union[float, np.ndarray], alpha_params: Dict[str, float], context: cl.Context = CONTEXT, queue: cl.CommandQueue = QUEUE) -> float:
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
        
        # Use the stored global and local sizes
        global_size = operational_efficiency_program.global_size
        local_size = operational_efficiency_program.local_size
        
        operational_efficiency_program.calculate_operational_efficiency(queue, global_size, local_size, P_buf, E_buf, alpha_buf, result_buf)
        
        result = np.empty_like(P)
        cl.enqueue_copy(queue, result, result_buf).wait()
        efficiency = np.sum(result)
        
        logging.debug(f"calculate_operational_efficiency: Calculated operational efficiency: {efficiency}")
        return efficiency
    except Exception as e:
        logging.error(f"calculate_operational_efficiency: Error calculating operational efficiency: {e}")
        raise

def calculate_error_management(error_detection_rate: Union[float, np.ndarray], correction_capability: Union[float, np.ndarray], alpha_params: Dict[str, float], context: cl.Context = CONTEXT, queue: cl.CommandQueue = QUEUE) -> float:
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
        
        # Use the stored global and local sizes
        global_size = error_management_program.global_size
        local_size = error_management_program.local_size
        
        error_management_program.calculate_error_management(queue, global_size, local_size, error_detection_rate_buf, correction_capability_buf, alpha_buf, result_buf)
        
        result = np.empty_like(error_detection_rate)
        cl.enqueue_copy(queue, result, result_buf).wait()
        error_management_value = float(np.sum(result))
        
        logging.debug(f"calculate_error_management: Calculated error management effectiveness: {error_management_value}")
        return error_management_value
    except Exception as e:
        logging.error(f"calculate_error_management: Error calculating error management effectiveness: {e}")
        raise

def calculate_adaptability(adaptation_rate: Union[float, np.ndarray], alpha_params: Dict[str, float], context: cl.Context = CONTEXT, queue: cl.CommandQueue = QUEUE) -> float:
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
        
        # Use the stored global and local sizes
        global_size = adaptability_program.global_size
        local_size = adaptability_program.local_size
        
        adaptability_program.calculate_adaptability(queue, global_size, local_size, adaptation_rate_buf, alpha_buf, result_buf)
        
        result = np.empty_like(adaptation_rate)
        cl.enqueue_copy(queue, result, result_buf).wait()
        adaptability_value = float(np.sum(result))
        
        logging.debug(f"calculate_adaptability: Calculated adaptability: {adaptability_value}")
        return adaptability_value
    except Exception as e:
        logging.error(f"calculate_adaptability: Error calculating adaptability: {e}")
        raise

def calculate_volume(spatial_scale: Union[float, np.ndarray], alpha_params: Dict[str, float], context: cl.Context = CONTEXT, queue: cl.CommandQueue = QUEUE) -> float:
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
        
        # Use the stored global and local sizes
        global_size = volume_program.global_size
        local_size = volume_program.local_size
        
        volume_program.calculate_volume(queue, global_size, local_size, spatial_scale_buf, alpha_buf, result_buf)
        
        result = np.empty_like(spatial_scale)
        cl.enqueue_copy(queue, result, result_buf).wait()
        volume_value = float(np.sum(result))
        
        logging.debug(f"calculate_volume: Calculated volume: {volume_value}")
        return volume_value
    except Exception as e:
        logging.error(f"calculate_volume: Error calculating volume: {e}")
        raise

def calculate_time(temporal_scale: Union[float, np.ndarray], alpha_params: Dict[str, float], context: cl.Context = CONTEXT, queue: cl.CommandQueue = QUEUE) -> float:
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
        
        # Use the stored global and local sizes
        global_size = time_program.global_size
        local_size = time_program.local_size
        
        time_program.calculate_time(queue, global_size, local_size, temporal_scale_buf, alpha_buf, result_buf)
        
        result = np.empty_like(temporal_scale)
        cl.enqueue_copy(queue, result, result_buf).wait()
        time_value = float(np.sum(result))
        
        logging.debug(f"calculate_time: Calculated time: {time_value}")
        return time_value
    except Exception as e:
        logging.error(f"calculate_time: Error calculating time: {e}")
        raise

def compute_intelligence(combination: Dict[str, float], context: cl.Context = CONTEXT, queue: cl.CommandQueue = QUEUE) -> Optional[float]:
    """
    Compute the intelligence metric for a given parameter combination, handling exceptions and ensuring robust computation.
    
    Parameters:
        combination (Dict[str, float]): The parameter combination.
        context (cl.Context): The OpenCL context.
        queue (cl.CommandQueue): The OpenCL command queue.
    
    Returns:
        Optional[float]: The computed intelligence metric or None if an error occurs.
    """
    logging.debug(f"Starting computation of intelligence metric with combination: {combination}")

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
        logging.debug(f"Extracted parameters: {extracted_params}")
        return extracted_params

    def validate_parameters(parameters: Dict[str, float]) -> None:
        """
        Validate the parameters to ensure non-zero and non-negligible denominators.
        
        Parameters:
            parameters (Dict[str, float]): The parameters to validate.
        
        Raises:
            ValueError: If Volume_value or Time_value are zero or negligible.
        """
        logging.debug(f"Validating parameters: {parameters}")
        if np.isclose(parameters["Volume_value"], 0) or np.isclose(parameters["Time_value"], 0):
            logging.error("Volume_value and Time_value must be non-zero and non-negligible.")
            raise ValueError("Volume_value and Time_value must be non-zero and non-negligible.")
        logging.debug("Parameter validation passed.")

    def compute_metric(parameters: Dict[str, float]) -> float:
        """
        Compute the intelligence metric using the provided parameters.
        
        Parameters:
            parameters (Dict[str, float]): The parameters for computation.
        
        Returns:
            float: The computed intelligence metric.
        """
        logging.debug(f"Computing intelligence metric with parameters: {parameters}")
        metric = parameters["k"] * (
            parameters["H_X_value"] * parameters["I_XY_value"] * parameters["O_value"] * 
            parameters["Em_value"] * parameters["A_value"]
        ) / (parameters["Volume_value"] * parameters["Time_value"])
        logging.debug(f"Computed intelligence metric: {metric}")
        return metric

    try:
        parameters: Dict[str, float] = extract_parameters(combination)
        validate_parameters(parameters)
        intelligence_metric: float = compute_metric(parameters)
        logging.debug(f"Computed intelligence metric: {intelligence_metric}")
        return intelligence_metric

    except ValueError as ve:
        logging.error(f"Validation error for combination {combination}: {ve}")
        return None
    except Exception as e:
        logging.error(f"Failed to compute intelligence for combination {combination}: {e}")
        return None

# Plotting Functions
def plot_heatmap(data: pd.DataFrame) -> None:
    """
    Plot the heatmap of the correlation matrix.

    Parameters:
        data (pd.DataFrame): The data for which the correlation matrix heatmap is to be plotted.
    """
    logging.debug(f"Starting to plot heatmap for data with shape {data.shape} and columns {data.columns.tolist()}")
    try:
        plt.figure(figsize=(14, 10))
        logging.debug("Figure created with size (14, 10)")
        correlation_matrix = data.corr()
        logging.debug(f"Correlation matrix calculated: {correlation_matrix}")
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        logging.debug("Heatmap created with 'coolwarm' colormap and linewidths=0.5")
        plt.title('Correlation Matrix of Parameters and Intelligence')
        plt.savefig('correlation_matrix.png')
        logging.debug("Heatmap saved as 'correlation_matrix.png'")
        plt.show(block=False)
        plt.pause(5)
        plt.close()
        logging.debug("Heatmap displayed and closed successfully.")
    except Exception as e:
        logging.error(f"Failed to plot heatmap: {e}")

def plot_pairplot(data: pd.DataFrame) -> None:
    """
    Plot the pairplot of the parameters.

    Parameters:
        data (pd.DataFrame): The data for which the pairplot is to be plotted.
    """
    logging.debug(f"Starting to plot pairplot for data with shape {data.shape} and columns {data.columns.tolist()}")
    try:
        sns.pairplot(data, hue='Intelligence', palette='viridis')
        logging.debug("Pairplot created with 'Intelligence' as hue and 'viridis' palette")
        plt.savefig('pairplot.png')
        logging.debug("Pairplot saved as 'pairplot.png'")
        plt.show(block=False)
        plt.pause(5)
        plt.close()
        logging.debug("Pairplot displayed and closed successfully.")
    except Exception as e:
        logging.error(f"Failed to plot pairplot: {e}")

def plot_jointplot(data: pd.DataFrame) -> None:
    """
    Plot the jointplot of the parameters.

    Parameters:
        data (pd.DataFrame): The data for which the jointplot is to be plotted.
    """
    logging.debug(f"Starting to plot jointplot for data with shape {data.shape} and columns {data.columns.tolist()}")
    try:
        sns.jointplot(data=data, x='H_X', y='Intelligence', kind='hex', cmap='Blues')
        logging.debug("Jointplot created with 'H_X' as x, 'Intelligence' as y, 'hex' kind, and 'Blues' colormap")
        plt.savefig('jointplot.png')
        logging.debug("Jointplot saved as 'jointplot.png'")
        plt.show(block=False)
        plt.pause(5)
        plt.close()
        logging.debug("Jointplot displayed and closed successfully.")
    except Exception as e:
        logging.error(f"Failed to plot jointplot: {e}")

def plot_histograms(data: pd.DataFrame) -> None:
    """
    Plot the histograms of the parameters.

    Parameters:
        data (pd.DataFrame): The data for which the histograms are to be plotted.
    """
    logging.debug(f"Starting to plot histograms for data with shape {data.shape} and columns {data.columns.tolist()}")
    try:
        for column in data.columns:
            logging.debug(f"Plotting histogram for column: {column}")
            plt.figure(figsize=(10, 6))
            sns.histplot(data[column], bins=20, kde=True, color='magenta')
            logging.debug(f"Histogram created for column {column} with 20 bins, KDE=True, and color 'magenta'")
            plt.title(f'Distribution of {column}')
            plt.savefig(f'{column}_distribution.png')
            logging.debug(f"Histogram for column {column} saved as '{column}_distribution.png'")
            plt.show(block=False)
            plt.pause(5)
            plt.close()
            logging.debug(f"Histogram for column {column} displayed and closed successfully.")
    except Exception as e:
        logging.error(f"Failed to plot histograms: {e}")

def plot_boxplots(data: pd.DataFrame) -> None:
    """
    Plot the boxplots of the parameters.

    Parameters:
        data (pd.DataFrame): The data for which the boxplots are to be plotted.
    """
    logging.debug(f"Starting to plot boxplots for data with shape {data.shape} and columns {data.columns.tolist()}")
    try:
        for column in data.columns[:-1]:
            logging.debug(f"Plotting boxplot for column: {column}")
            plt.figure(figsize=(12, 8))
            sns.boxplot(x='Intelligence', y=column, data=data)
            logging.debug(f"Boxplot created for column {column} with 'Intelligence' as x-axis")
            plt.title(f'Intelligence vs {column}')
            plt.savefig(f'Intelligence_vs_{column}.png')
            logging.debug(f"Boxplot for column {column} saved as 'Intelligence_vs_{column}.png'")
            plt.show(block=False)
            plt.pause(5)
            plt.close()
            logging.debug(f"Boxplot for column {column} displayed and closed successfully.")
    except Exception as e:
        logging.error(f"Failed to plot boxplots: {e}")

def plot_violinplot(data: pd.DataFrame) -> None:
    """
    Plot the violinplot of the parameters.

    Parameters:
        data (pd.DataFrame): The data for which the violinplot is to be plotted.
    """
    logging.debug(f"Starting to plot violinplot for data with shape {data.shape} and columns {data.columns.tolist()}")
    try:
        plt.figure(figsize=(12, 8))
        sns.violinplot(data=data, x='Error Detection Rate', y='Correction Capability', scale='width', inner='quartile')
        logging.debug("Violinplot created with 'Error Detection Rate' as x, 'Correction Capability' as y, scale='width', and inner='quartile'")
        plt.title('Error Detection Rate vs Correction Capability')
        plt.savefig('Error_Detection_vs_Correction_Capability_violin.png')
        logging.debug("Violinplot saved as 'Error_Detection_vs_Correction_Capability_violin.png'")
        plt.show(block=False)
        plt.pause(5)
        plt.close()
        logging.debug("Violinplot displayed and closed successfully.")
    except Exception as e:
        logging.error(f"Failed to plot violinplot: {e}")

def plot_scatter_matrix(data: pd.DataFrame) -> None:
    """
    Plot the scatter matrix of the parameters.

    Parameters:
        data (pd.DataFrame): The data for which the scatter matrix is to be plotted.
    """
    logging.debug(f"Starting to plot scatter matrix for data with shape {data.shape} and columns {data.columns.tolist()}")
    try:
        plt.figure(figsize=(20, 15))
        pd.plotting.scatter_matrix(data, alpha=0.8, figsize=(20, 15), diagonal='kde')
        logging.debug("Scatter matrix created with alpha=0.8 and diagonal='kde'")
        plt.savefig('scatter_matrix.png')
        logging.debug("Scatter matrix saved as 'scatter_matrix.png'")
        plt.show(block=False)
        plt.pause(5)
        plt.close()
        logging.debug("Scatter matrix displayed and closed successfully.")
    except Exception as e:
        logging.error(f"Failed to plot scatter matrix: {e}")

def plot_ridge_plot(data: pd.DataFrame) -> None:
    """
    Plot the ridge plot of the parameters.

    Parameters:
        data (pd.DataFrame): The data for which the ridge plot is to be plotted.
    """
    logging.debug(f"Starting to plot ridge plot for data with shape {data.shape} and columns {data.columns.tolist()}")
    try:
        fig, axes = joypy.joyplot(data, by='Intelligence', figsize=(12, 8), colormap=plt.cm.viridis, alpha=0.8)
        logging.debug("Ridge plot created with 'Intelligence' as by, figsize=(12, 8), colormap=plt.cm.viridis, and alpha=0.8")
        plt.title('Ridge Plot of Parameters Grouped by Intelligence')
        plt.savefig('ridge_plot.png')
        logging.debug("Ridge plot saved as 'ridge_plot.png'")
        plt.show(block=False)
        plt.pause(5)
        plt.close()
        logging.debug("Ridge plot displayed and closed successfully.")
    except Exception as e:
        logging.error(f"Failed to plot ridge plot: {e}")

def plot_3d_scatter(data: pd.DataFrame) -> None:
    """
    Plot the 3D scatter plot of the parameters.

    Parameters:
        data (pd.DataFrame): The data for which the 3D scatter plot is to be plotted.
    """
    logging.debug(f"Starting to plot 3D scatter plot for data with shape {data.shape} and columns {data.columns.tolist()}")
    try:
        fig = plt.figure(figsize=(14, 10))
        logging.debug("3D scatter plot figure created with size (14, 10)")
        ax = fig.add_subplot(111, projection='3d')
        logging.debug("3D scatter plot axis created with projection='3d'")
        ax.scatter(data['H_X'], data['P'], data['Intelligence'], c='r', marker='o')
        logging.debug("3D scatter plot created with 'H_X' as x, 'P' as y, 'Intelligence' as z, color='r', and marker='o'")
        ax.set_xlabel('H_X')
        ax.set_ylabel('P')
        ax.set_zlabel('Intelligence')
        plt.title('3D Scatter Plot of H_X, P and Intelligence')
        plt.savefig('3D_scatter_H_X_P_Intelligence.png')
        logging.debug("3D scatter plot saved as '3D_scatter_H_X_P_Intelligence.png'")
        plt.show(block=False)
        plt.pause(5)
        plt.close()
        logging.debug("3D scatter plot displayed and closed successfully.")
    except Exception as e:
        logging.error(f"Failed to plot 3D scatter: {e}")

def visualize_results(results: pd.DataFrame) -> None:
    """
    Visualize the results using various plots.

    Parameters:
        results (pd.DataFrame): The results DataFrame.
    """
    logging.debug(f"Starting to visualize results with shape {results.shape} and columns {results.columns.tolist()}")
    try:
        numeric_results = results.drop(columns=['Probabilities'])
        logging.debug(f"Dropped 'Probabilities' column, resulting in numeric results with shape {numeric_results.shape} and columns {numeric_results.columns.tolist()}")
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

# Main execution
def main() -> None:
    logging.info("Starting Universal Intelligence Potential simulation.")
    try:
        opencl_manager = OpenCLManager()
        parameter_manager = ParameterManager(opencl_manager)
        total_operations = parameter_manager.calculate_total_operations()
        progress_bar.total = total_operations
        combinations = parameter_manager.generate_combinations()

        results = []
        for combination in tqdm(combinations, desc="Processing Combinations", unit="combination"):
            intelligence_metric = compute_intelligence(combination, opencl_manager)
            if intelligence_metric is not None:
                results.append({**combination, "Intelligence": intelligence_metric})

        results_df = pd.DataFrame(results)
        results_df.to_csv("simulation_results.csv", index=False)
        visualize_results(results_df)

    except Exception as e:
        logging.error(f"An error occurred during the simulation: {e}")
        raise

    finally:
        logging.info("Simulation completed.")

if __name__ == "__main__":
    main()
