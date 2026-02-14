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

# Configure logging to capture debug information
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Define alpha parameters
alpha_parameters: Dict[str, float] = {
    "k": 1.0, "alpha_H": 1.0, "alpha_I": 1.0, "alpha_O": 1.0, "alpha_Em": 1.0,
    "alpha_A": 1.0, "alpha_Volume": 1.0, "alpha_t": 1.0, "alpha_Pi": 1.0,
    "alpha_log": 1.0, "alpha_HX": 1.0, "alpha_HY": 1.0, "alpha_HXY": 1.0,
    "alpha_P": 1.0, "alpha_E": 1.0, "alpha_Error_Detection": 1.0,
    "alpha_Correction": 1.0, "alpha_Adaptation_Rate": 1.0, "alpha_Spatial_Scale": 1.0,
    "alpha_Temporal_Scale": 1.0
}

class OpenCLManager:
    """
    Manages OpenCL environment and programs.
    """
    def __init__(self, kernel_directory: str = "/home/lloyd/UniversalIntelligencePotential/kernels/"):
        """
        Initializes the OpenCL manager with the specified kernel directory.

        Args:
            kernel_directory (str): The directory containing OpenCL kernels.
        """
        self.kernel_directory = kernel_directory
        self.context, self.queue = self.setup_opencl_environment()
        self.programs = self.load_all_programs()

    def setup_opencl_environment(self) -> Tuple[cl.Context, cl.CommandQueue]:
        """
        Sets up the OpenCL environment, creating a context and command queue.

        Returns:
            Tuple[cl.Context, cl.CommandQueue]: The OpenCL context and command queue.

        Raises:
            Exception: If an error occurs during OpenCL environment setup.
        """
        try:
            logging.debug("Attempting to get OpenCL platforms.")
            platform = cl.get_platforms()[0]
            logging.debug(f"Platform obtained: {platform}")
            
            logging.debug("Attempting to get OpenCL devices from platform.")
            device = platform.get_devices()[0]
            logging.debug(f"Device obtained: {device}")
            
            logging.debug("Creating OpenCL context with the obtained device.")
            context = cl.Context([device])
            logging.debug("OpenCL context created successfully.")
            
            logging.debug("Creating OpenCL command queue with the created context.")
            queue = cl.CommandQueue(context)
            logging.debug("OpenCL command queue created successfully.")
            
            return context, queue
        except Exception as e:
            logging.error(f"Error setting up OpenCL environment: {e}")
            raise

    def load_kernel(self, filename: str) -> str:
        """
        Loads the OpenCL kernel code from the specified file.

        Args:
            filename (str): The name of the kernel file.

        Returns:
            str: The OpenCL kernel code.

        Raises:
            IOError: If an error occurs while opening the kernel file.
            Exception: If an unexpected error occurs while loading the kernel.
        """
        try:
            logging.debug(f"Attempting to open kernel file: {filename}")
            with open(filename, 'r') as file:
                kernel_code = file.read()
            logging.debug(f"Kernel loaded successfully from {filename}")
            return kernel_code
        except IOError as e:
            logging.error(f"Error opening kernel file {filename}: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error loading kernel from {filename}: {e}")
            raise

    def load_and_build_program(self, kernel_path: str) -> cl.Program:
        """
        Loads the OpenCL kernel code from the specified path and builds the program.

        Args:
            kernel_path (str): The path to the kernel file.

        Returns:
            cl.Program: The built OpenCL program.

        Raises:
            Exception: If an error occurs while loading or building the program.
        """
        try:
            logging.debug(f"Loading kernel from path: {kernel_path}")
            kernel_code = self.load_kernel(kernel_path)
            
            logging.debug(f"Building OpenCL program from kernel code.")
            program = cl.Program(self.context, kernel_code).build()
            logging.debug(f"Program built successfully from {kernel_path}")
            
            return program
        except Exception as e:
            logging.error(f"Failed to load and build program from {kernel_path}: {e}")
            raise

    def load_all_programs(self) -> Dict[str, cl.Program]:
        """
        Loads and builds all OpenCL programs from the kernel directory.

        Returns:
            Dict[str, cl.Program]: A dictionary mapping program names to OpenCL programs.
        """
        kernel_files = [
            'entropy_kernel.cl', 'mutual_information_kernel.cl', 'operational_efficiency_kernel.cl',
            'error_management_kernel.cl', 'adaptability_kernel.cl', 'volume_kernel.cl', 'time_kernel.cl'
        ]
        programs = {}
        for kernel_file in kernel_files:
            program_name = kernel_file.split('_')[0]
            programs[program_name] = self.load_and_build_program(os.path.join(self.kernel_directory, kernel_file))
        return programs

class ParameterManager:
    """
    Manages parameters and their ranges.
    """
    def __init__(self, parameter_file: str = 'parameter_ranges.npz', alpha_params: Optional[Dict[str, float]] = None):
        """
        Initializes the Parameter Manager with the specified parameter file and alpha parameters.

        Args:
            parameter_file (str): The file containing parameter ranges.
            alpha_params (Optional[Dict[str, float]]): Alpha parameters to use, defaults to global alpha_parameters.
        """
        self.parameter_file = parameter_file
        self.alpha_params = alpha_params if alpha_params else alpha_parameters
        self.parameter_ranges = self.load_parameter_ranges()
        self.total_operations = self.calculate_total_operations()
        self.progress_bar = tqdm(total=self.total_operations, desc="Overall Progress", unit="operation")

    def load_parameter_ranges(self) -> Dict[str, np.ndarray]:
        """
        Loads parameter ranges from the specified file or creates and saves them if the file does not exist.

        Returns:
            Dict[str, np.ndarray]: The loaded or created parameter ranges.

        Raises:
            Exception: If an error occurs while loading or creating parameter ranges.
        """
        if os.path.exists(self.parameter_file):
            try:
                logging.debug(f"Loading parameter ranges from {self.parameter_file}")
                with np.load(self.parameter_file, allow_pickle=True) as data:
                    parameter_ranges = {key: data[key] for key in data}
                logging.debug("Parameter ranges loaded successfully.")
                return parameter_ranges
            except Exception as e:
                logging.error(f"Error loading parameter ranges from {self.parameter_file}: {e}")
                raise
        else:
            return self.create_and_save_parameter_ranges()

    def create_and_save_parameter_ranges(self) -> Dict[str, np.ndarray]:
        """
        Creates and saves parameter ranges to the specified file.

        Returns:
            Dict[str, np.ndarray]: The created parameter ranges.

        Raises:
            Exception: If an error occurs while creating or saving parameter ranges.
        """
        try:
            logging.debug("Creating parameter ranges.")
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
            logging.debug(f"Saving parameter ranges to {self.parameter_file}")
            np.savez(self.parameter_file, **parameter_ranges)
            logging.debug("Parameter ranges saved successfully.")
            return parameter_ranges
        except Exception as e:
            logging.error(f"Error creating or saving parameter ranges: {e}")
            raise

    def generate_probabilities_set(self) -> Dict[str, np.ndarray]:
        """
        Generates a set of probability distributions.

        Returns:
            Dict[str, np.ndarray]: The generated probability distributions.
        """
        logging.debug("Generating probabilities set.")
        probabilities_set = {
            f'prob_{p1}_{p2}': np.array([p1, p2, 1 - p1 - p2])
            for p1 in np.linspace(0.1, 0.9, 9)
            for p2 in np.linspace(0.1, 0.9, 9)
            if p1 + p2 <= 1
        }
        logging.debug("Probabilities set generated successfully.")
        return probabilities_set

    def calculate_total_operations(self) -> int:
        """
        Calculates the total number of operations based on parameter combinations.

        Returns:
            int: The total number of operations.
        """
        logging.debug("Calculating total operations.")
        combinations = self.generate_combinations()
        total_operations = len(combinations)
        logging.debug(f"Total operations calculated: {total_operations}")
        return total_operations

    def generate_combinations(self) -> List[Dict[str, Any]]:
        """
        Generates all possible combinations of parameter values.

        Returns:
            List[Dict[str, Any]]: The generated parameter combinations.
        """
        logging.debug("Generating parameter combinations.")
        keys = list(self.parameter_ranges.keys())
        values = [self.parameter_ranges[key] for key in keys]
        combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
        logging.debug("Parameter combinations generated successfully.")
        return combinations

def calculate_entropy(probabilities: Union[np.ndarray, List[float]], parameter_manager: ParameterManager, opencl_manager: OpenCLManager) -> float:
    """
    Calculates the Shannon entropy using OpenCL.

    Args:
        probabilities (Union[np.ndarray, List[float]]): The probability distribution.
        parameter_manager (ParameterManager): The parameter manager.
        opencl_manager (OpenCLManager): The OpenCL manager.

    Returns:
        float: The calculated Shannon entropy.

    Raises:
        Exception: If an error occurs during the calculation.
    """
    try:
        probabilities = np.array(probabilities, dtype=np.float32)
        alpha_values = np.array([parameter_manager.alpha_params["alpha_H"], parameter_manager.alpha_params["alpha_Pi"], parameter_manager.alpha_params["alpha_log"]], dtype=np.float32)
        prob_buf = cl.Buffer(opencl_manager.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=probabilities)
        alpha_buf = cl.Buffer(opencl_manager.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=alpha_values)
        result_buf = cl.Buffer(opencl_manager.context, cl.mem_flags.WRITE_ONLY, probabilities.nbytes)
        opencl_manager.programs['entropy'].calculate_entropy(opencl_manager.queue, probabilities.shape, None, prob_buf, alpha_buf, result_buf, np.int32(len(probabilities)))
        result = np.empty_like(probabilities)
        cl.enqueue_copy(opencl_manager.queue, result, result_buf).wait()
        entropy = np.sum(result)
        logging.debug(f"calculate_entropy: Calculated Shannon entropy: {entropy}")
        return entropy
    except Exception as e:
        logging.error(f"calculate_entropy: Error calculating Shannon entropy: {e}")
        raise

def calculate_mutual_information(H_X: Union[float, np.ndarray], H_Y: Union[float, np.ndarray], H_XY: Union[float, np.ndarray], parameter_manager: ParameterManager, opencl_manager: OpenCLManager) -> float:
    """
    Calculates the mutual information using OpenCL.

    Args:
        H_X (Union[float, np.ndarray]): The entropy of X.
        H_Y (Union[float, np.ndarray]): The entropy of Y.
        H_XY (Union[float, np.ndarray]): The joint entropy of X and Y.
        parameter_manager (ParameterManager): The parameter manager.
        opencl_manager (OpenCLManager): The OpenCL manager.

    Returns:
        float: The calculated mutual information.

    Raises:
        Exception: If an error occurs during the calculation.
    """
    try:
        H_X = np.array(H_X, dtype=np.float32)
        H_Y = np.array(H_Y, dtype=np.float32)
        H_XY = np.array(H_XY, dtype=np.float32)
        alpha_values = np.array([parameter_manager.alpha_params["alpha_I"], parameter_manager.alpha_params["alpha_HX"], parameter_manager.alpha_params["alpha_HY"], parameter_manager.alpha_params["alpha_HXY"]], dtype=np.float32)
        H_X_buf = cl.Buffer(opencl_manager.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=H_X)
        H_Y_buf = cl.Buffer(opencl_manager.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=H_Y)
        H_XY_buf = cl.Buffer(opencl_manager.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=H_XY)
        alpha_buf = cl.Buffer(opencl_manager.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=alpha_values)
        result_buf = cl.Buffer(opencl_manager.context, cl.mem_flags.WRITE_ONLY, H_X.nbytes)
        opencl_manager.programs['mutual_information'].calculate_mutual_information(opencl_manager.queue, H_X.shape, None, H_X_buf, H_Y_buf, H_XY_buf, alpha_buf, result_buf)
        result = np.empty_like(H_X)
        cl.enqueue_copy(opencl_manager.queue, result, result_buf).wait()
        mutual_info = np.sum(result)
        logging.debug(f"calculate_mutual_information: Calculated mutual information: {mutual_info}")
        return mutual_info
    except Exception as e:
        logging.error(f"calculate_mutual_information: Error calculating mutual information: {e}")
        raise

def calculate_operational_efficiency(P: Union[float, np.ndarray], E: Union[float, np.ndarray], parameter_manager: ParameterManager, opencl_manager: OpenCLManager) -> float:
    """
    Calculates the operational efficiency using OpenCL.

    Args:
        P (Union[float, np.ndarray]): The performance metric.
        E (Union[float, np.ndarray]): The efficiency metric.
        parameter_manager (ParameterManager): The parameter manager.
        opencl_manager (OpenCLManager): The OpenCL manager.

    Returns:
        float: The calculated operational efficiency.

    Raises:
        Exception: If an error occurs during the calculation.
    """
    try:
        P = np.asarray(P, dtype=np.float32)
        E = np.asarray(E, dtype=np.float32)
        alpha_values = np.array([parameter_manager.alpha_params["alpha_O"], parameter_manager.alpha_params["alpha_P"], parameter_manager.alpha_params["alpha_E"]], dtype=np.float32)
        P_buf = cl.Buffer(opencl_manager.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=P)
        E_buf = cl.Buffer(opencl_manager.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=E)
        alpha_buf = cl.Buffer(opencl_manager.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=alpha_values)
        result_buf = cl.Buffer(opencl_manager.context, cl.mem_flags.WRITE_ONLY, P.nbytes)
        opencl_manager.programs['operational_efficiency'].calculate_operational_efficiency(opencl_manager.queue, P.shape, None, P_buf, E_buf, alpha_buf, result_buf)
        result = np.empty_like(P)
        cl.enqueue_copy(opencl_manager.queue, result, result_buf).wait()
        efficiency = np.sum(result)
        logging.debug(f"calculate_operational_efficiency: Calculated operational efficiency: {efficiency}")
        return efficiency
    except Exception as e:
        logging.error(f"calculate_operational_efficiency: Error calculating operational efficiency: {e}")
        raise

def calculate_error_management(error_detection_rate: Union[float, np.ndarray], correction_capability: Union[float, np.ndarray], parameter_manager: ParameterManager, opencl_manager: OpenCLManager) -> float:
    """
    Calculates the error management effectiveness using OpenCL.

    Args:
        error_detection_rate (Union[float, np.ndarray]): The error detection rate.
        correction_capability (Union[float, np.ndarray]): The correction capability.
        parameter_manager (ParameterManager): The parameter manager.
        opencl_manager (OpenCLManager): The OpenCL manager.

    Returns:
        float: The calculated error management effectiveness.

    Raises:
        Exception: If an error occurs during the calculation.
    """
    try:
        error_detection_rate = np.asarray(error_detection_rate, dtype=np.float32)
        correction_capability = np.asarray(correction_capability, dtype=np.float32)
        alpha_values = np.array([parameter_manager.alpha_params["alpha_Em"], parameter_manager.alpha_params["alpha_Error_Detection"], parameter_manager.alpha_params["alpha_Correction"]], dtype=np.float32)
        error_detection_rate_buf = cl.Buffer(opencl_manager.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=error_detection_rate)
        correction_capability_buf = cl.Buffer(opencl_manager.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=correction_capability)
        alpha_buf = cl.Buffer(opencl_manager.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=alpha_values)
        result_buf = cl.Buffer(opencl_manager.context, cl.mem_flags.WRITE_ONLY, error_detection_rate.nbytes)
        opencl_manager.programs['error_management'].calculate_error_management(opencl_manager.queue, error_detection_rate.shape, None, error_detection_rate_buf, correction_capability_buf, alpha_buf, result_buf)
        result = np.empty_like(error_detection_rate)
        cl.enqueue_copy(opencl_manager.queue, result, result_buf).wait()
        error_management_value = float(np.sum(result))
        logging.debug(f"calculate_error_management: Calculated error management effectiveness: {error_management_value}")
        return error_management_value
    except Exception as e:
        logging.error(f"calculate_error_management: Error calculating error management effectiveness: {e}")
        raise

def calculate_adaptability(adaptation_rate: Union[float, np.ndarray], parameter_manager: ParameterManager, opencl_manager: OpenCLManager) -> float:
    try:
        adaptation_rate = np.asarray(adaptation_rate, dtype=np.float32)
        alpha_values = np.array([parameter_manager.alpha_params["alpha_A"], parameter_manager.alpha_params["alpha_Adaptation_Rate"]], dtype=np.float32)
        adaptation_rate_buf = cl.Buffer(opencl_manager.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=adaptation_rate)
        alpha_buf = cl.Buffer(opencl_manager.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=alpha_values)
        result_buf = cl.Buffer(opencl_manager.context, cl.mem_flags.WRITE_ONLY, adaptation_rate.nbytes)
        opencl_manager.programs['adaptability'].calculate_adaptability(opencl_manager.queue, adaptation_rate.shape, None, adaptation_rate_buf, alpha_buf, result_buf)
        result = np.empty_like(adaptation_rate)
        cl.enqueue_copy(opencl_manager.queue, result, result_buf).wait()
        adaptability_value = float(np.sum(result))
        logging.debug(f"calculate_adaptability: Calculated adaptability: {adaptability_value}")
        return adaptability_value
    except Exception as e:
        logging.error(f"calculate_adaptability: Error calculating adaptability: {e}")
        raise

def calculate_volume(spatial_scale: Union[float, np.ndarray], parameter_manager: ParameterManager, opencl_manager: OpenCLManager) -> float:
    try:
        spatial_scale = np.asarray(spatial_scale, dtype=np.float32)
        alpha_values = np.array([parameter_manager.alpha_params["alpha_Volume"], parameter_manager.alpha_params["alpha_Spatial_Scale"]], dtype=np.float32)
        spatial_scale_buf = cl.Buffer(opencl_manager.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=spatial_scale)
        alpha_buf = cl.Buffer(opencl_manager.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=alpha_values)
        result_buf = cl.Buffer(opencl_manager.context, cl.mem_flags.WRITE_ONLY, spatial_scale.nbytes)
        opencl_manager.programs['volume'].calculate_volume(opencl_manager.queue, spatial_scale.shape, None, spatial_scale_buf, alpha_buf, result_buf)
        result = np.empty_like(spatial_scale)
        cl.enqueue_copy(opencl_manager.queue, result, result_buf).wait()
        volume_value = float(np.sum(result))
        logging.debug(f"calculate_volume: Calculated volume: {volume_value}")
        return volume_value
    except Exception as e:
        logging.error(f"calculate_volume: Error calculating volume: {e}")
        raise

def calculate_time(temporal_scale: Union[float, np.ndarray], parameter_manager: ParameterManager, opencl_manager: OpenCLManager) -> float:
    try:
        temporal_scale = np.asarray(temporal_scale, dtype=np.float32)
        alpha_values = np.array([parameter_manager.alpha_params["alpha_t"], parameter_manager.alpha_params["alpha_Temporal_Scale"]], dtype=np.float32)
        temporal_scale_buf = cl.Buffer(opencl_manager.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=temporal_scale)
        alpha_buf = cl.Buffer(opencl_manager.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=alpha_values)
        result_buf = cl.Buffer(opencl_manager.context, cl.mem_flags.WRITE_ONLY, temporal_scale.nbytes)
        opencl_manager.programs['time'].calculate_time(opencl_manager.queue, temporal_scale.shape, None, temporal_scale_buf, alpha_buf, result_buf)
        result = np.empty_like(temporal_scale)
        cl.enqueue_copy(opencl_manager.queue, result, result_buf).wait()
        time_value = float(np.sum(result))
        logging.debug(f"calculate_time: Calculated time: {time_value}")
        return time_value
    except Exception as e:
        logging.error(f"calculate_time: Error calculating time: {e}")
        raise

def compute_intelligence_v2(combination: Dict[str, float], parameter_manager: ParameterManager, opencl_manager: OpenCLManager) -> Optional[float]:
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

def main() -> None:
    logging.info("Starting Universal Intelligence Potential simulation.")
    try:
        # Initialize OpenCL Manager
        opencl_manager = OpenCLManager()

        # Initialize Parameter Manager
        parameter_manager = ParameterManager()

        # Generate parameter combinations
        combinations = parameter_manager.generate_combinations()

        # List to store results
        results = []

        # Iterate over all combinations
        for combination in tqdm(combinations, desc="Processing Combinations", unit="combination"):
            # Compute intelligence metric for the combination
            intelligence_metric = compute_intelligence_v2(combination, parameter_manager, opencl_manager)
            if intelligence_metric is not None:
                results.append({**combination, "Intelligence": intelligence_metric})

        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        results_df.to_csv("simulation_results.csv", index=False)

        # Visualize results
        visualize_results(results_df)

    except Exception as e:
        logging.error(f"An error occurred during the simulation: {e}")
        raise
    finally:
        logging.info("Simulation completed.")

if __name__ == "__main__":
    main()
