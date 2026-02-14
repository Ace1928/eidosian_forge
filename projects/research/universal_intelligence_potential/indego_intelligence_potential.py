import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Dict, Any, Union
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joypy
from tqdm import tqdm
import pyopencl as cl
from mpl_toolkits.mplot3d import Axes3D

# Configure logging with detailed format and handlers
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

# Define the parameters for the model with explicit type annotations and detailed logging
alpha_parameters: Dict[str, float] = {
    "k": 1.0, "alpha_H": 1.0, "alpha_I": 1.0, "alpha_O": 1.0, "alpha_Em": 1.0,
    "alpha_A": 1.0, "alpha_Volume": 1.0, "alpha_t": 1.0, "alpha_Pi": 1.0,
    "alpha_log": 1.0, "alpha_HX": 1.0, "alpha_HY": 1.0, "alpha_HXY": 1.0,
    "alpha_P": 1.0, "alpha_E": 1.0, "alpha_Error_Detection": 1.0,
    "alpha_Correction": 1.0, "alpha_Adaptation_Rate": 1.0, "alpha_Spatial_Scale": 1.0,
    "alpha_Temporal_Scale": 1.0
}
logging.debug(f"Alpha parameters initialized: {alpha_parameters}")

def validate_imports() -> None:
    """
    Validates the imports of required modules.
    Logs detailed information about each module's import status.
    Raises ImportError if any module is not imported correctly.
    """
    logging.debug("Starting import validation process.")
    modules: Dict[str, Any] = {
        'numpy': np,
        'pandas': pd,
        'logging': logging,
        'itertools': itertools,
        'matplotlib.pyplot': plt,
        'seaborn': sns,
        'os': os,
        'joypy': joypy,
        'tqdm': tqdm,
        'pyopencl': cl,
        'mpl_toolkits.mplot3d': Axes3D
    }

    for module_name, module in modules.items():
        try:
            logging.debug(f"Validating import for module: {module_name}")
            if module is None:
                raise ImportError(f"Module {module_name} is not imported correctly.")
            logging.info(f"Module {module_name} imported successfully.")
        except ImportError as ie:
            logging.critical(f"Failed to import module {module_name}: {ie}", exc_info=True)
            raise
        except Exception as e:
            logging.critical(f"Unexpected error during import validation for module {module_name}: {e}", exc_info=True)
            raise

try:
    logging.debug("Attempting to validate imports.")
    validate_imports()
    logging.debug("Import validation completed successfully.")
except ImportError as ie:
    logging.critical(f"Import validation failed due to ImportError: {ie}", exc_info=True)
    raise
except Exception as e:
    logging.critical(f"Import validation failed due to an unexpected error: {e}", exc_info=True)
    raise

class OpenCLManager:
    def __init__(self):
        """
        Initializes the OpenCLManager by setting up the OpenCL environment, clearing the build cache,
        and loading all OpenCL programs. Logs detailed information about each step and handles errors comprehensively.
        """
        logging.debug("Initializing OpenCLManager.")
        try:
            self.context, self.queue = self.setup_opencl_environment()
            logging.debug(f"OpenCL context: {self.context}, OpenCL queue: {self.queue}")
            self.clear_build_cache()
            self.programs = self.load_all_programs()
            logging.debug(f"Loaded OpenCL programs: {self.programs}")
        except Exception as e:
            logging.critical(f"Failed to initialize OpenCLManager: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize OpenCLManager: {e}")

    def setup_opencl_environment(self) -> Tuple[cl.Context, cl.CommandQueue]:
        """
        Sets up the OpenCL environment by selecting the first platform and its devices, creating a context and a command queue.
        Logs detailed information about each step and handles errors comprehensively.

        Returns:
            Tuple[cl.Context, cl.CommandQueue]: The created OpenCL context and command queue.
        """
        logging.debug("Setting up OpenCL environment.")
        try:
            platform = cl.get_platforms()[0]
            logging.debug(f"Selected OpenCL platform: {platform}")
            devices = platform.get_devices()
            logging.debug(f"OpenCL devices: {devices}")
            context = cl.Context(devices)
            queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)
            logging.info("OpenCL environment setup successfully.")
            return context, queue
        except cl.Error as cl_err:
            logging.error(f"OpenCL error during environment setup: {cl_err}", exc_info=True)
            raise RuntimeError(f"OpenCL error during environment setup: {cl_err}")
        except Exception as e:
            logging.error(f"Unexpected error setting up OpenCL environment: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error setting up OpenCL environment: {e}")

    def clear_build_cache(self):
        """
        Clears the OpenCL build cache by building an empty program.
        Logs detailed information about each step and handles errors comprehensively.
        """
        logging.debug("Clearing OpenCL build cache.")
        try:
            cl.Program(self.context, "").build()
            logging.info("OpenCL build cache cleared successfully.")
        except cl.Error as cl_err:
            logging.error(f"OpenCL error during build cache clearing: {cl_err}", exc_info=True)
            raise RuntimeError(f"OpenCL error during build cache clearing: {cl_err}")
        except Exception as e:
            logging.error(f"Unexpected error during build cache clearing: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error during build cache clearing: {e}")

    def load_kernel(self, filename: str) -> str:
        """
        Loads the kernel code from a file.
        Logs detailed information about each step and handles errors comprehensively.

        Args:
            filename (str): The path to the kernel file.

        Returns:
            str: The kernel code as a string.
        """
        logging.debug(f"Loading kernel from file: {filename}")
        try:
            with open(filename, 'r') as file:
                kernel_code = file.read()
                logging.info(f"Kernel loaded from {filename}.")
                logging.debug(f"Kernel code: {kernel_code[:100]}...")  # Log the first 100 characters for brevity
                return kernel_code
        except FileNotFoundError as fnf_error:
            logging.error(f"Kernel file not found: {filename}. Error: {fnf_error}", exc_info=True)
            raise
        except IOError as io_error:
            logging.error(f"IOError while reading kernel file {filename}: {io_error}", exc_info=True)
            raise
        except Exception as e:
            logging.error(f"Unexpected error loading kernel from {filename}: {e}", exc_info=True)
            raise

    def load_and_build_program(self, kernel_path: str) -> cl.Program:
        """
        Loads and builds an OpenCL program from a kernel file.
        Logs detailed information about each step and handles errors comprehensively.

        Args:
            kernel_path (str): The path to the kernel file.

        Returns:
            cl.Program: The built OpenCL program.
        """
        logging.debug(f"Loading and building program from kernel path: {kernel_path}")
        try:
            kernel_code = self.load_kernel(kernel_path)
            logging.debug(f"Kernel code loaded for building: {kernel_code[:100]}...")  # Log the first 100 characters for brevity
            program = cl.Program(self.context, kernel_code).build()
            logging.info(f"Program built successfully from {kernel_path}.")
            return program
        except FileNotFoundError as fnf_error:
            logging.error(f"Kernel file not found: {kernel_path}. Error: {fnf_error}", exc_info=True)
            raise
        except cl.Error as cl_error:
            logging.error(f"OpenCL error while building program from {kernel_path}: {cl_error}", exc_info=True)
            raise
        except Exception as e:
            logging.error(f"Unexpected error while loading and building program from {kernel_path}: {e}", exc_info=True)
            raise

    def load_all_programs(self) -> Dict[str, cl.Program]:
        """
        Loads and builds all OpenCL programs from predefined kernel files.
        Logs detailed information about each step and handles errors comprehensively.

        Returns:
            Dict[str, cl.Program]: A dictionary of program names to their corresponding OpenCL programs.
        """
        logging.debug("Loading all OpenCL programs.")
        kernel_directory = "/home/lloyd/UniversalIntelligencePotential/kernels/"
        kernel_files = [
            'entropy_kernel.cl', 'mutual_information_kernel.cl', 'operational_efficiency_kernel.cl',
            'error_management_kernel.cl', 'adaptability_kernel.cl', 'volume_kernel.cl', 'time_kernel.cl'
        ]
        programs = {}

        for kernel_file in kernel_files:
            try:
                program_name = kernel_file.split('_')[0]
                kernel_path = os.path.join(kernel_directory, kernel_file)
                logging.debug(f"Loading and building program: {program_name} from {kernel_path}")
                programs[program_name] = self.load_and_build_program(kernel_path)
                logging.info(f"Program {program_name} loaded and built successfully from {kernel_path}.")
            except FileNotFoundError as fnf_error:
                logging.error(f"Kernel file {kernel_file} not found: {fnf_error}", exc_info=True)
                raise
            except cl.Error as cl_error:
                logging.error(f"OpenCL error while building program from {kernel_file}: {cl_error}", exc_info=True)
                raise
            except Exception as e:
                logging.error(f"Unexpected error while loading and building program from {kernel_file}: {e}", exc_info=True)
                raise

        logging.debug(f"All programs loaded: {programs}")
        return programs

class ParameterManager:
    def __init__(self):
        try:
            logging.debug("Initializing ParameterManager.")
            self.parameter_ranges = self.load_parameter_ranges()
            logging.info("ParameterManager initialized successfully with parameter_ranges: %s", self.parameter_ranges)
            self.alpha_parameters: Dict[str, float] = alpha_parameters
            logging.debug("Alpha parameters set: %s", self.alpha_parameters)
        except ValueError as ve:
            logging.critical(f"ValueError occurred while initializing ParameterManager: {ve}", exc_info=True)
            raise
        except TypeError as te:
            logging.critical(f"TypeError occurred while initializing ParameterManager: {te}", exc_info=True)
            raise
        except KeyError as ke:
            logging.critical(f"KeyError occurred while initializing ParameterManager: {ke}", exc_info=True)
            raise
        except Exception as e:
            logging.critical(f"Unexpected error occurred while initializing ParameterManager: {e}", exc_info=True)
            raise

    def generate_probabilities_set(self) -> Dict[str, np.ndarray]:
        probabilities_set: Dict[str, np.ndarray] = {}
        try:
            logging.debug("Starting to generate probabilities set.")
            for p1 in np.linspace(0.1, 0.9, 3):
                logging.debug(f"Generating probabilities for p1: {p1}")
                for p2 in np.linspace(0.1, 0.9, 3):
                    logging.debug(f"Generating probabilities for p2: {p2}")
                    if p1 + p2 <= 1:
                        key = f'prob_{p1:.1f}_{p2:.1f}'
                        value = np.array([p1, p2, 1 - p1 - p2], dtype=float)
                        probabilities_set[key] = value
                        logging.debug(f"Generated probability set {key}: {value}")
            logging.info("Probabilities set generated successfully: %s", probabilities_set)
        except ValueError as ve:
            logging.error(f"ValueError occurred while generating probabilities set: {ve}", exc_info=True)
            raise
        except TypeError as te:
            logging.error(f"TypeError occurred while generating probabilities set: {te}", exc_info=True)
            raise
        except Exception as e:
            logging.error(f"Unexpected error occurred while generating probabilities set: {e}", exc_info=True)
            raise
        return probabilities_set

    def load_parameter_ranges(self) -> Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]:
        try:
            logging.debug("Starting to load parameter ranges.")
            
            if os.path.exists('parameter_ranges.npz'):
                logging.debug("parameter_ranges.npz file exists. Loading from file.")
                with np.load('parameter_ranges.npz', allow_pickle=True) as data:
                    parameter_ranges = {
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
                    logging.info("Parameter ranges loaded from file successfully: %s", parameter_ranges)
            else:
                logging.debug("parameter_ranges.npz file does not exist. Generating new parameter ranges.")
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
                logging.info("Parameter ranges saved to file successfully: %s", parameter_ranges)

            for key, value in parameter_ranges.items():
                logging.debug(f"Validating parameter range for key: {key}, value: {value}")
                if not isinstance(value, (np.ndarray, dict)):
                    error_message = f"Parameter range for '{key}' is not a numpy array or dictionary."
                    logging.error(error_message)
                    raise ValueError(error_message)

            logging.info("Parameter ranges loaded and validated successfully: %s", parameter_ranges)
            return parameter_ranges

        except ValueError as ve:
            logging.error(f"ValueError occurred while loading parameter ranges: {ve}", exc_info=True)
            raise
        except Exception as e:
            logging.error(f"Unexpected error occurred while loading parameter ranges: {e}", exc_info=True)
            raise
    
    def generate_combinations(self, parameter_ranges: Dict[str, Union[np.ndarray, float]]) -> List[Dict[str, Union[np.ndarray, float]]]:
        logging.debug("Starting to generate combinations with parameter_ranges: %s", parameter_ranges)
        self.parameter_ranges = parameter_ranges
        try:
            if not isinstance(self.parameter_ranges, dict):
                error_message = "The parameter_ranges attribute must be a dictionary."
                logging.error(error_message)
                raise ValueError(error_message)

            if not self.parameter_ranges:
                error_message = "The parameter_ranges attribute is empty."
                logging.error(error_message)
                raise ValueError(error_message)

            keys: List[str] = list(self.parameter_ranges.keys())
            logging.debug("Parameter keys: %s", keys)
            values: List[Union[np.ndarray, float]] = [self.parameter_ranges[key] for key in keys]
            logging.debug("Parameter values: %s", values)

            combinations: List[Dict[str, Union[np.ndarray, float]]] = [
                dict(zip(keys, combination)) for combination in itertools.product(*values)
            ]

            logging.info("Parameter combinations generated successfully: %s", combinations)
            return combinations

        except ValueError as ve:
            logging.error(f"ValueError occurred: {ve}", exc_info=True)
            raise
        except Exception as e:
            logging.error(f"Error generating parameter combinations: {e}", exc_info=True)
            raise

class KernelManager:
    def __init__(self, opencl_manager: OpenCLManager):
        logging.debug(f"Initializing KernelManager with opencl_manager: {opencl_manager}")
        self.opencl_manager = opencl_manager
        self.parameter_manager = ParameterManager()
        logging.debug(f"ParameterManager initialized: {self.parameter_manager}")
        self.programs: Dict[str, cl.Program] = {}
        self.context, self.queue = self.opencl_manager.setup_opencl_environment()
        logging.debug(f"OpenCL environment setup with context: {self.context}, queue: {self.queue}")
        self.programs = self.opencl_manager.load_all_programs()
        logging.debug(f"Loaded OpenCL programs: {self.programs}")

    def calculate_entropy(self, probabilities: np.ndarray, alpha_parameters: Dict[str, float] = alpha_parameters) -> float:
        logging.debug(f"calculate_entropy called with probabilities: {probabilities}, alpha_parameters: {alpha_parameters}")
        try:
            probabilities = np.asarray(probabilities, dtype=np.float32)
            logging.debug(f"Converted probabilities to np.ndarray with dtype=np.float32: {probabilities}")
            alpha_H = alpha_parameters["alpha_H"]
            alpha_Pi = alpha_parameters["alpha_Pi"]
            alpha_log = alpha_parameters["alpha_log"]
            logging.debug(f"Extracted alpha parameters: alpha_H={alpha_H}, alpha_Pi={alpha_Pi}, alpha_log={alpha_log}")
            prob_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=probabilities)
            logging.debug(f"Created OpenCL buffer for probabilities: {prob_buf}")
            alpha_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.array([alpha_H, alpha_Pi, alpha_log], dtype=np.float32))
            logging.debug(f"Created OpenCL buffer for alpha parameters: {alpha_buf}")
            result_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, probabilities.nbytes)
            logging.debug(f"Created OpenCL buffer for result: {result_buf}")
            self.programs['entropy'].calculate_entropy(
                self.queue, 
                (probabilities.size,), 
                None, 
                prob_buf, 
                alpha_buf, 
                result_buf,
                np.int32(probabilities.size)
            )
            logging.debug(f"Enqueued kernel execution for entropy calculation with queue: {self.queue}, size: {probabilities.size}")
            result = np.empty_like(probabilities)
            cl.enqueue_copy(self.queue, result, result_buf).wait()
            logging.debug(f"Copied result from OpenCL buffer to host: {result}")
            entropy_value = float(np.sum(result))
            logging.debug(f"Calculated entropy value: {entropy_value}")
            return entropy_value
        except Exception as e:
            logging.error(f"calculate_entropy: Error calculating entropy: {e}", exc_info=True)
            raise
        
    def calculate_mutual_information(self, H_X: np.ndarray, H_Y: np.ndarray, H_XY: np.ndarray) -> float:
        logging.debug(f"calculate_mutual_information called with H_X: {H_X}, H_Y: {H_Y}, H_XY: {H_XY}")
        try:
            H_X = np.asarray(H_X, dtype=np.float32)
            H_Y = np.asarray(H_Y, dtype=np.float32)
            H_XY = np.asarray(H_XY, dtype=np.float32)
            logging.debug(f"Converted H_X, H_Y, H_XY to np.ndarray with dtype=np.float32")
            H_X_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=H_X)
            H_Y_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=H_Y)
            H_XY_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=H_XY)
            logging.debug(f"Created OpenCL buffers for H_X: {H_X_buf}, H_Y: {H_Y_buf}, H_XY: {H_XY_buf}")
            result_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, H_X.nbytes)
            logging.debug(f"Created OpenCL buffer for result: {result_buf}")
            self.programs['mutual_information'].calculate_mutual_information(self.queue, H_X.shape, None, H_X_buf, H_Y_buf, H_XY_buf, result_buf)
            logging.debug(f"Enqueued kernel execution for mutual information calculation with queue: {self.queue}, shape: {H_X.shape}")
            result = np.empty_like(H_X)
            cl.enqueue_copy(self.queue, result, result_buf).wait()
            logging.debug(f"Copied result from OpenCL buffer to host: {result}")
            mutual_info = float(np.sum(result))
            logging.debug(f"Calculated mutual information value: {mutual_info}")
            return mutual_info
        except Exception as e:
            logging.error(f"calculate_mutual_information: Error calculating mutual information: {e}", exc_info=True)
            raise

    def calculate_operational_efficiency(self, P: np.ndarray, E: np.ndarray) -> float:
        logging.debug(f"calculate_operational_efficiency called with P: {P}, E: {E}")
        try:
            P = np.asarray(P, dtype=np.float32)
            E = np.asarray(E, dtype=np.float32)
            logging.debug(f"Converted P and E to np.ndarray with dtype=np.float32")
            P_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=P)
            E_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=E)
            logging.debug(f"Created OpenCL buffers for P: {P_buf}, E: {E_buf}")
            result_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, P.nbytes)
            logging.debug(f"Created OpenCL buffer for result: {result_buf}")
            self.programs['operational_efficiency'].calculate_operational_efficiency(self.queue, P.shape, None, P_buf, E_buf, result_buf)
            logging.debug(f"Enqueued kernel execution for operational efficiency calculation with queue: {self.queue}, shape: {P.shape}")
            result = np.empty_like(P)
            cl.enqueue_copy(self.queue, result, result_buf).wait()
            logging.debug(f"Copied result from OpenCL buffer to host: {result}")
            efficiency = float(np.sum(result))
            logging.debug(f"Calculated operational efficiency value: {efficiency}")
            return efficiency
        except Exception as e:
            logging.error(f"calculate_operational_efficiency: Error calculating operational efficiency: {e}", exc_info=True)
            raise

    def calculate_error_management(self, error_detection_rate: np.ndarray, correction_capability: np.ndarray) -> float:
        logging.debug(f"calculate_error_management called with error_detection_rate: {error_detection_rate}, correction_capability: {correction_capability}")
        try:
            error_detection_rate = np.asarray(error_detection_rate, dtype=np.float32)
            correction_capability = np.asarray(correction_capability, dtype=np.float32)
            logging.debug(f"Converted error_detection_rate and correction_capability to np.ndarray with dtype=np.float32")
            error_detection_rate_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=error_detection_rate)
            correction_capability_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=correction_capability)
            logging.debug(f"Created OpenCL buffers for error_detection_rate: {error_detection_rate_buf}, correction_capability: {correction_capability_buf}")
            result_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, error_detection_rate.nbytes)
            logging.debug(f"Created OpenCL buffer for result: {result_buf}")
            self.programs['error_management'].calculate_error_management(self.queue, error_detection_rate.shape, None, error_detection_rate_buf, correction_capability_buf, result_buf)
            logging.debug(f"Enqueued kernel execution for error management calculation with queue: {self.queue}, shape: {error_detection_rate.shape}")
            result = np.empty_like(error_detection_rate)
            cl.enqueue_copy(self.queue, result, result_buf).wait()
            logging.debug(f"Copied result from OpenCL buffer to host: {result}")
            error_management_value = float(np.sum(result))
            logging.debug(f"Calculated error management effectiveness value: {error_management_value}")
            return error_management_value
        except Exception as e:
            logging.error(f"calculate_error_management: Error calculating error management effectiveness: {e}", exc_info=True)
            raise

    def calculate_adaptability(self, adaptation_rate: np.ndarray) -> float:
        logging.debug(f"calculate_adaptability called with adaptation_rate: {adaptation_rate}")
        try:
            adaptation_rate = np.asarray(adaptation_rate, dtype=np.float32)
            logging.debug(f"Converted adaptation_rate to np.ndarray with dtype=np.float32: {adaptation_rate}")
            adaptation_rate_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=adaptation_rate)
            logging.debug(f"Created OpenCL buffer for adaptation_rate: {adaptation_rate_buf}")
            result_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, adaptation_rate.nbytes)
            logging.debug(f"Created OpenCL buffer for result: {result_buf}")
            self.programs['adaptability'].calculate_adaptability(self.queue, adaptation_rate.shape, None, adaptation_rate_buf, result_buf)
            logging.debug(f"Enqueued kernel execution for adaptability calculation with queue: {self.queue}, shape: {adaptation_rate.shape}")
            result = np.empty_like(adaptation_rate)
            cl.enqueue_copy(self.queue, result, result_buf).wait()
            logging.debug(f"Copied result from OpenCL buffer to host: {result}")
            adaptability_value = float(np.sum(result))
            logging.debug(f"Calculated adaptability value: {adaptability_value}")
            return adaptability_value
        except Exception as e:
            logging.error(f"calculate_adaptability: Error calculating adaptability: {e}", exc_info=True)
            raise

    def calculate_volume(self, spatial_scale: np.ndarray) -> float:
        logging.debug(f"calculate_volume called with spatial_scale: {spatial_scale}")
        try:
            spatial_scale = np.asarray(spatial_scale, dtype=np.float32)
            logging.debug(f"Converted spatial_scale to np.ndarray with dtype=np.float32: {spatial_scale}")
            spatial_scale_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=spatial_scale)
            logging.debug(f"Created OpenCL buffer for spatial_scale: {spatial_scale_buf}")
            result_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, spatial_scale.nbytes)
            logging.debug(f"Created OpenCL buffer for result: {result_buf}")
            self.programs['volume'].calculate_volume(self.queue, spatial_scale.shape, None, spatial_scale_buf, result_buf)
            logging.debug(f"Enqueued kernel execution for volume calculation with queue: {self.queue}, shape: {spatial_scale.shape}")
            result = np.empty_like(spatial_scale)
            cl.enqueue_copy(self.queue, result, result_buf).wait()
            logging.debug(f"Copied result from OpenCL buffer to host: {result}")
            volume_value = float(np.sum(result))
            logging.debug(f"Calculated volume value: {volume_value}")
            return volume_value
        except Exception as e:
            logging.error(f"calculate_volume: Error calculating volume: {e}", exc_info=True)
            raise

    def calculate_time(self, temporal_scale: np.ndarray) -> float:
        logging.debug(f"calculate_time called with temporal_scale: {temporal_scale}")
        try:
            temporal_scale = np.asarray(temporal_scale, dtype=np.float32)
            logging.debug(f"Converted temporal_scale to np.ndarray with dtype=np.float32: {temporal_scale}")
            temporal_scale_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=temporal_scale)
            logging.debug(f"Created OpenCL buffer for temporal_scale: {temporal_scale_buf}")
            result_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, temporal_scale.nbytes)
            logging.debug(f"Created OpenCL buffer for result: {result_buf}")
            self.programs['time'].calculate_time(self.queue, temporal_scale.shape, None, temporal_scale_buf, result_buf)
            logging.debug(f"Enqueued kernel execution for time calculation with queue: {self.queue}, shape: {temporal_scale.shape}")
            result = np.empty_like(temporal_scale)
            cl.enqueue_copy(self.queue, result, result_buf).wait()
            logging.debug(f"Copied result from OpenCL buffer to host: {result}")
            time_value = float(np.sum(result))
            logging.debug(f"Calculated time value: {time_value}")
            return time_value
        except Exception as e:
            logging.error(f"calculate_time: Error calculating time: {e}", exc_info=True)
            raise

class DataVisualizer:
    def plot_heatmap(self):
        logging.debug("Starting to plot heatmap.")
        try:
            logging.debug("Creating figure with size (14, 10).")
            plt.figure(figsize=(14, 10))
            correlation_matrix = self.data.corr()
            logging.debug(f"Calculated correlation matrix: {correlation_matrix}")
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
            plt.title('Correlation Matrix of Parameters and Intelligence')
            logging.debug("Saving heatmap as 'correlation_matrix.png'.")
            plt.savefig('correlation_matrix.png')
            plt.show()
            logging.info("Heatmap plotted and saved successfully.")
        except Exception as e:
            logging.error(f"Failed to plot heatmap: {e}", exc_info=True)
            raise

    def plot_pairplot(self, hue='Intelligence'):
        logging.debug(f"Starting to plot pairplot with hue {hue}.")
        try:
            logging.debug(f"Creating pairplot with hue={hue} and palette='viridis'.")
            sns.pairplot(self.data, hue=hue, palette='viridis')
            logging.debug("Saving pairplot as 'pairplot.png'.")
            plt.savefig('pairplot.png')
            plt.show()
            logging.info("Pairplot plotted and saved successfully.")
        except Exception as e:
            logging.error(f"Failed to plot pairplot: {e}", exc_info=True)
            raise

    def plot_jointplot(self, x='H_X', y='Intelligence'):
        logging.debug(f"Starting to plot jointplot with x={x} and y={y}.")
        
        if x not in self.data.columns:
            error_message = f"The column '{x}' does not exist in the dataset."
            logging.error(error_message)
            raise ValueError(error_message)
        
        if y not in self.data.columns:
            error_message = f"The column '{y}' does not exist in the dataset."
            logging.error(error_message)
            raise ValueError(error_message)
        
        try:
            logging.debug(f"Creating jointplot with x={x}, y={y}, kind='hex', cmap='Blues'.")
            sns.jointplot(data=self.data, x=x, y=y, kind='hex', cmap='Blues')
            logging.debug("Saving jointplot as 'jointplot.png'.")
            plt.savefig('jointplot.png')
            plt.show()
            logging.info("Jointplot plotted and saved successfully.")
        except Exception as e:
            logging.error(f"Failed to plot jointplot: {e}", exc_info=True)
            raise

    def plot_histograms(self):
        logging.debug("Starting to plot histograms.")
        
        if not isinstance(self.data, pd.DataFrame):
            error_message = "The dataset is not a pandas DataFrame."
            logging.error(error_message)
            raise ValueError(error_message)
        
        if self.data.empty:
            error_message = "The dataset is empty."
            logging.error(error_message)
            raise ValueError(error_message)
        
        try:
            for column in self.data.columns:
                logging.debug(f"Plotting histogram for column: {column}")
                plt.figure(figsize=(10, 6))
                sns.histplot(self.data[column], bins=20, kde=True, color='magenta')
                plt.title(f'Distribution of {column}')
                filename = f'{column}_distribution.png'
                logging.debug(f"Saving histogram for column {column} as {filename}.")
                plt.savefig(filename)
                logging.info(f"Histogram for column {column} saved as {filename}")
                plt.show()
        except ValueError as ve:
            logging.error(f"ValueError encountered while plotting histograms: {ve}", exc_info=True)
            raise
        except TypeError as te:
            logging.error(f"TypeError encountered while plotting histograms: {te}", exc_info=True)
            raise
        except Exception as e:
            logging.error(f"Failed to plot histograms due to an unexpected error: {e}", exc_info=True)
            raise

    def plot_boxplots(self, x='Intelligence'):
        logging.debug(f"Starting to plot boxplots with x={x}.")

        if x not in self.data.columns:
            error_message = f"The specified x column '{x}' is not in the dataset."
            logging.error(error_message)
            raise ValueError(error_message)

        try:
            for column in self.data.columns:
                if column != x:
                    logging.debug(f"Plotting boxplot for column: {column}")
                    plt.figure(figsize=(12, 8))
                    sns.boxplot(x=x, y=column, data=self.data)
                    plt.title(f'{x} vs {column}')
                    filename = f'{x}_vs_{column}.png'
                    logging.debug(f"Saving boxplot for {x} vs {column} as '{filename}'.")
                    plt.savefig(filename)
                    logging.info(f"Boxplot saved as '{filename}'.")
                    plt.show()
                    logging.info(f"Boxplot for {x} vs {column} displayed successfully.")
        except Exception as e:
            logging.error(f"Failed to plot boxplots: {e}", exc_info=True)
            raise

    def plot_violinplot(self, x, y):
        logging.debug(f"Starting to plot violinplot with x={x} and y={y}.")

        if x not in self.data.columns:
            error_message = f"The column '{x}' does not exist in the dataset."
            logging.error(error_message)
            raise ValueError(error_message)
        
        if y not in self.data.columns:
            error_message = f"The column '{y}' does not exist in the dataset."
            logging.error(error_message)
            raise ValueError(error_message)

        try:
            logging.debug(f"Creating figure with size (12, 8) for violinplot.")
            plt.figure(figsize=(12, 8))
            logging.debug(f"Creating violinplot with x={x}, y={y}, scale='width', inner='quartile'.")
            sns.violinplot(data=self.data, x=x, y=y, scale='width', inner='quartile')
            plt.title(f'{x} vs {y}')
            filename = f'{x}_vs_{y}_violin.png'
            logging.debug(f"Saving violin plot as {filename}.")
            plt.savefig(filename)
            plt.show()
            logging.info(f"Violin plot for {x} vs {y} created and saved successfully.")
        except Exception as e:
            logging.error(f"Failed to plot violinplot: {e}", exc_info=True)
            raise

    def plot_scatter_matrix(self):
        logging.debug("Starting to plot scatter matrix.")
        
        if not isinstance(self.data, pd.DataFrame):
            error_message = "The dataset is not a pandas DataFrame."
            logging.error(error_message)
            raise ValueError(error_message)
        
        if self.data.empty:
            error_message = "The dataset is empty."
            logging.error(error_message)
            raise ValueError(error_message)
        
        try:
            logging.debug("Creating figure with size (20, 15) for scatter matrix.")
            plt.figure(figsize=(20, 15))
            logging.debug("Creating scatter matrix plot with alpha=0.8 and diagonal='kde'.")
            scatter_matrix = pd.plotting.scatter_matrix(self.data, alpha=0.8, figsize=(20, 15), diagonal='kde')
            logging.debug("Saving scatter matrix plot as 'scatter_matrix.png'.")
            plt.savefig('scatter_matrix.png')
            logging.info("Scatter matrix plot saved as 'scatter_matrix.png'.")
            plt.show()
            logging.info("Scatter matrix plot displayed successfully.")
        except Exception as e:
            logging.error(f"Failed to plot scatter matrix: {e}", exc_info=True)
            raise

    def plot_ridge_plot(self, by='Intelligence'):
        logging.debug(f"Starting to plot ridge plot grouped by {by}.")
        
        if by not in self.data.columns:
            error_message = f"The specified column '{by}' for grouping does not exist in the dataset."
            logging.error(error_message)
            raise ValueError(error_message)
        
        try:
            logging.debug(f"Creating ridge plot with by={by}, figsize=(12, 8), colormap=plt.cm.viridis, alpha=0.8.")
            fig, axes = joypy.joyplot(
                self.data, 
                by=by, 
                figsize=(12, 8), 
                colormap=plt.cm.viridis, 
                alpha=0.8
            )
            plt.title(f'Ridge Plot of Parameters Grouped by {by}')
            logging.debug("Saving ridge plot as 'ridge_plot.png'.")
            plt.savefig('ridge_plot.png')
            plt.show()
            logging.info(f"Ridge plot grouped by {by} plotted successfully.")
        except Exception as e:
            logging.error(f"Failed to plot ridge plot: {e}", exc_info=True)
            raise

    def plot_3d_scatter(self, x='H_X', y='P', z='Intelligence'):
        logging.debug(f"Starting to plot 3D scatter plot with x={x}, y={y}, and z={z}.")
        
        for col in [x, y, z]:
            if col not in self.data.columns:
                error_message = f"Column '{col}' not found in the dataset."
                logging.error(error_message)
                raise ValueError(error_message)
        
        try:
            logging.debug("Creating figure with size (14, 10) for 3D scatter plot.")
            fig = plt.figure(figsize=(14, 10))
            logging.debug("Adding 3D subplot.")
            ax = fig.add_subplot(111, projection='3d')
            logging.debug(f"Creating 3D scatter plot with x={x}, y={y}, z={z}, color='r', marker='o'.")
            ax.scatter(self.data[x], self.data[y], self.data[z], c='r', marker='o')
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.set_zlabel(z)
            plt.title(f'3D Scatter Plot of {x}, {y} and {z}')
            plot_filename = '3D_scatter_plot.png'
            logging.debug(f"Saving 3D scatter plot as {plot_filename}.")
            plt.savefig(plot_filename)
            logging.info(f"3D scatter plot saved as {plot_filename}")
            plt.show()
            logging.debug("3D scatter plot displayed successfully.")
        
        except ValueError as ve:
            logging.error(f"ValueError occurred: {ve}", exc_info=True)
            raise
        
        except Exception as e:
            logging.error(f"Failed to plot 3D scatter: {e}", exc_info=True)
            raise

    def visualize_results(self, data):
        logging.debug("Starting to visualize results.")
        self.data = data
        try:
            if not isinstance(self.data, pd.DataFrame):
                error_message = "The data attribute must be a pandas DataFrame."
                logging.error(error_message)
                raise ValueError(error_message)
            
            numeric_results = self.data.select_dtypes(include=[np.number])
            logging.debug(f"Numeric columns selected for visualization: {numeric_results.columns.tolist()}")

            try:
                logging.debug("Calling plot_heatmap method.")
                self.plot_heatmap()
                logging.info("Heatmap plotted successfully.")
            except Exception as e:
                logging.error(f"Failed to plot heatmap: {e}", exc_info=True)
                raise

            try:
                logging.debug("Calling plot_pairplot method.")
                self.plot_pairplot()
                logging.info("Pairplot plotted successfully.")
            except Exception as e:
                logging.error(f"Failed to plot pairplot: {e}", exc_info=True)
                raise

            try:
                logging.debug("Calling plot_jointplot method.")
                self.plot_jointplot()
                logging.info("Jointplot plotted successfully.")
            except Exception as e:
                logging.error(f"Failed to plot jointplot: {e}", exc_info=True)
                raise

            try:
                logging.debug("Calling plot_histograms method.")
                self.plot_histograms()
                logging.info("Histograms plotted successfully.")
            except Exception as e:
                logging.error(f"Failed to plot histograms: {e}", exc_info=True)
                raise

            try:
                logging.debug("Calling plot_boxplots method.")
                self.plot_boxplots()
                logging.info("Boxplots plotted successfully.")
            except Exception as e:
                logging.error(f"Failed to plot boxplots: {e}", exc_info=True)
                raise

            try:
                logging.debug("Calling plot_violinplot method with 'Error Detection Rate' and 'Correction Capability'.")
                self.plot_violinplot('Error Detection Rate', 'Correction Capability')
                logging.info("Violinplot plotted successfully.")
            except Exception as e:
                logging.error(f"Failed to plot violinplot: {e}", exc_info=True)
                raise

            try:
                logging.debug("Calling plot_scatter_matrix method.")
                self.plot_scatter_matrix()
                logging.info("Scatter matrix plotted successfully.")
            except Exception as e:
                logging.error(f"Failed to plot scatter matrix: {e}", exc_info=True)
                raise

            try:
                logging.debug("Calling plot_ridge_plot method.")
                self.plot_ridge_plot()
                logging.info("Ridge plot plotted successfully.")
            except Exception as e:
                logging.error(f"Failed to plot ridge plot: {e}", exc_info=True)
                raise

            try:
                logging.debug("Calling plot_3d_scatter method.")
                self.plot_3d_scatter()
                logging.info("3D scatter plot plotted successfully.")
            except Exception as e:
                logging.error(f"Failed to plot 3D scatter: {e}", exc_info=True)
                raise
        
        except ValueError as ve:
            logging.error(f"ValueError during visualization: {ve}", exc_info=True)
            raise
        except Exception as e:
            logging.error(f"Failed to visualize results: {e}", exc_info=True)
            raise
        finally:
            logging.info("Visualization process completed.")

def main():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.debug("Logging initialized with DEBUG level and specified format.")
    
    logging.debug("Instantiating OpenCLManager.")
    opencl_manager = OpenCLManager()
    logging.debug(f"OpenCLManager instantiated: {opencl_manager}")
    
    logging.debug("Instantiating KernelManager with OpenCLManager.")
    kernel_manager = KernelManager(opencl_manager)
    logging.debug(f"KernelManager instantiated: {kernel_manager}")
    
    logging.debug("Instantiating DataVisualizer.")
    data_visualizer = DataVisualizer()
    logging.debug(f"DataVisualizer instantiated: {data_visualizer}")
    
    logging.debug("Instantiating ParameterManager.")
    parameter_manager = ParameterManager()
    logging.debug(f"ParameterManager instantiated: {parameter_manager}")

    # Check if parameter ranges file exists and load it if available
    if os.path.exists('parameter_ranges.npz'):
        logging.debug("Parameter ranges file 'parameter_ranges.npz' exists. Loading data.")
        with np.load('parameter_ranges.npz', allow_pickle=True) as data:
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
            logging.debug("Parameter ranges loaded successfully from 'parameter_ranges.npz'.")
    else:
        logging.debug("Parameter ranges file 'parameter_ranges.npz' does not exist. Generating new parameter ranges.")
        # Define entropy and other parameters with refined granularity
        probabilities_set = np.linspace(0.0, 1.0, 20)
        H_X_set = np.linspace(0.2, 2.0, 3)
        H_Y_set = np.linspace(0.2, 2.0, 3)
        H_XY_set = np.linspace(0.5, 3.0, 3)
        P_set = np.linspace(100.0, 2000.0, 3)
        E_set = np.linspace(20.0, 100.0, 3)
        error_detection_rate_set = np.linspace(0.5, 1.0, 3)
        correction_capability_set = np.linspace(0.5, 1.0, 3)
        adaptation_rate_set = np.linspace(0.3, 1.0, 3)
        spatial_scale_set = np.linspace(0.5, 1.5, 3)
        temporal_scale_set = np.linspace(0.5, 1.5, 3)
        
        logging.debug(f"Generated parameter ranges: H_X_set={H_X_set}, H_Y_set={H_Y_set}, H_XY_set={H_XY_set}, P_set={P_set}, E_set={E_set}, error_detection_rate_set={error_detection_rate_set}, correction_capability_set={correction_capability_set}, adaptation_rate_set={adaptation_rate_set}, spatial_scale_set={spatial_scale_set}, temporal_scale_set={temporal_scale_set}")

        # Save the refined parameter ranges to a file for future use
        np.savez('parameter_ranges.npz', probabilities_set=probabilities_set, H_X_set=H_X_set, H_Y_set=H_Y_set, H_XY_set=H_XY_set,
                 P_set=P_set, E_set=E_set, error_detection_rate_set=error_detection_rate_set,
                 correction_capability_set=correction_capability_set, adaptation_rate_set=adaptation_rate_set,
                 spatial_scale_set=spatial_scale_set, temporal_scale_set=temporal_scale_set)
        logging.info("Parameter ranges saved to 'parameter_ranges.npz'.")

    # Prepare a DataFrame with enhanced structure to store the results
    results = pd.DataFrame(columns=[
        'Probabilities', 'H_X', 'H_Y', 'H_XY', 'P', 'E', 'Error Detection Rate',
        'Correction Capability', 'Adaptation Rate', 'Spatial Scale', 'Temporal Scale', 'Intelligence'
    ])
    logging.debug(f"Initialized results DataFrame with columns: {results.columns.tolist()}")

    # Log the shapes of the arrays to debug
    logging.debug(f"probabilities_set shape: {probabilities_set.shape}")
    logging.debug(f"H_X_set shape: {H_X_set.shape}")
    logging.debug(f"H_Y_set shape: {H_Y_set.shape}")
    logging.debug(f"H_XY_set shape: {H_XY_set.shape}")
    logging.debug(f"P_set shape: {P_set.shape}")
    logging.debug(f"E_set shape: {E_set.shape}")
    logging.debug(f"error_detection_rate_set shape: {error_detection_rate_set.shape}")
    logging.debug(f"correction_capability_set shape: {correction_capability_set.shape}")
    logging.debug(f"adaptation_rate_set shape: {adaptation_rate_set.shape}")
    logging.debug(f"spatial_scale_set shape: {spatial_scale_set.shape}")
    logging.debug(f"temporal_scale_set shape: {temporal_scale_set.shape}")

    # Ensure all arrays are not zero-dimensional
    arrays = [
        probabilities_set, H_X_set, H_Y_set, H_XY_set, P_set, E_set,
        error_detection_rate_set, correction_capability_set, adaptation_rate_set,
        spatial_scale_set, temporal_scale_set
    ]
    
    for array in arrays:
        if array.ndim == 0:
            raise ValueError(f"Array {array} is zero-dimensional and cannot be iterated over.")

    # Generate all combinations
    all_combinations = list(itertools.product(
        probabilities_set, H_X_set, H_Y_set, H_XY_set, P_set, E_set,
        error_detection_rate_set, correction_capability_set, adaptation_rate_set,
        spatial_scale_set, temporal_scale_set
    ))

    logging.debug(f"Generated {len(all_combinations)} combinations.")
    # Process each combination with enhanced computational methods
    for combination in all_combinations:
        probabilities, H_X, H_Y, H_XY, P, E, error_detection_rate, correction_capability, adaptation_rate, spatial_scale, temporal_scale = combination
        logging.debug(f"Processing combination: probabilities={probabilities}, H_X={H_X}, H_Y={H_Y}, H_XY={H_XY}, P={P}, E={E}, error_detection_rate={error_detection_rate}, correction_capability={correction_capability}, adaptation_rate={adaptation_rate}, spatial_scale={spatial_scale}, temporal_scale={temporal_scale}")

        # Calculate each component with refined algorithms
        H_X_value = kernel_manager.calculate_entropy(probabilities)
        logging.debug(f"Calculated H_X_value: {H_X_value}")
        
        I_XY_value = kernel_manager.calculate_mutual_information(H_X, H_Y, H_XY)
        logging.debug(f"Calculated I_XY_value: {I_XY_value}")
        
        O_value = kernel_manager.calculate_operational_efficiency(P, E)
        logging.debug(f"Calculated O_value: {O_value}")
        
        Em_value = kernel_manager.calculate_error_management(error_detection_rate, correction_capability)
        logging.debug(f"Calculated Em_value: {Em_value}")
        
        A_value = kernel_manager.calculate_adaptability(adaptation_rate)
        logging.debug(f"Calculated A_value: {A_value}")
        
        Volume_value = kernel_manager.calculate_volume(spatial_scale)
        logging.debug(f"Calculated Volume_value: {Volume_value}")
        
        Time_value = kernel_manager.calculate_time(temporal_scale)
        logging.debug(f"Calculated Time_value: {Time_value}")

        # Compute intelligence with an optimized formula
        I = parameter_manager.alpha_parameters["k"] * (H_X_value * I_XY_value * O_value * Em_value * A_value) / (Volume_value * Time_value)
        logging.debug(f"Computed intelligence value: {I}")

        # Append results to the DataFrame with enhanced data handling
        results = pd.concat([results, pd.DataFrame({
            'Probabilities': [probabilities], 'H_X': [H_X], 'H_Y': [H_Y], 'H_XY': [H_XY],
            'P': [P], 'E': [E], 'Error Detection Rate': [error_detection_rate],
            'Correction Capability': [correction_capability], 'Adaptation Rate': [adaptation_rate],
            'Spatial Scale': [spatial_scale], 'Temporal Scale': [temporal_scale], 'Intelligence': [I]
        })], ignore_index=True)
        logging.debug("Appended new row to results DataFrame.")

    # Output the results as a formatted table
    logging.debug("Outputting results DataFrame as a formatted table.")
    print(results.to_string(index=False))
    
    # Visualize the results
    logging.debug("Calling visualize_results method of DataVisualizer.")
    data_visualizer.visualize_results(results)
    logging.info("Visualization process completed successfully.")

if __name__ == "__main__":
    main()

