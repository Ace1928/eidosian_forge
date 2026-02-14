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

# Initialize progress bar for overall process
progress_bar = tqdm(total=100, desc="Overall Progress", unit="operation")

class OpenCLManager:
    def __init__(self):
        self.context, self.queue = self.setup_opencl_environment()
        self.programs = self.load_all_programs()

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

    def load_and_build_program(self, kernel_path: str) -> cl.Program:
        try:
            kernel_code = self.load_kernel(kernel_path)
            program = cl.Program(self.context, kernel_code).build()
            return program
        except Exception as e:
            logging.error(f"Failed to load and build program from {kernel_path}: {e}")
            raise

    def load_all_programs(self) -> Dict[str, cl.Program]:
        kernel_directory = "/home/lloyd/UniversalIntelligencePotential/kernels/"
        kernel_files = [
            'entropy_kernel.cl', 'mutual_information_kernel.cl', 'operational_efficiency_kernel.cl',
            'error_management_kernel.cl', 'adaptability_kernel.cl', 'volume_kernel.cl', 'time_kernel.cl'
        ]
        programs = {}
        for kernel_file in kernel_files:
            program_name = kernel_file.split('_')[0]
            programs[program_name] = self.load_and_build_program(kernel_directory + kernel_file)
        return programs

class ParameterManager:
    def __init__(self):
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

def execute_kernel(program: cl.Program, kernel_name: str, args: List, context: cl.Context, queue: cl.CommandQueue) -> np.ndarray:
    """
    Execute an OpenCL kernel with the provided arguments.

    Parameters:
        program (cl.Program): The OpenCL program containing the kernel.
        kernel_name (str): The name of the kernel to execute.
        args (List): The arguments to pass to the kernel.
        context (cl.Context): The OpenCL context.
        queue (cl.CommandQueue): The OpenCL command queue.

    Returns:
        np.ndarray: The result of the kernel execution.

    Raises:
        Exception: If an error occurs during kernel execution.
    """
    try:
        kernel = cl.Kernel(program, kernel_name)
        for i, arg in enumerate(args):
            kernel.set_arg(i, arg)

        result_buf = args[-1]
        global_size = (result_buf.size // np.dtype(np.float32).itemsize,)
        local_size = None

        cl.enqueue_nd_range_kernel(queue, kernel, global_size, local_size).wait()
        result = np.empty(global_size, dtype=np.float32)
        cl.enqueue_copy(queue, result, result_buf).wait()
        return result
    except Exception as e:
        logging.error(f"Error executing kernel {kernel_name}: {e}")
        raise


def calculate_metric(program: cl.Program, kernel_name: str, params: np.ndarray, alpha_values: np.ndarray, context: cl.Context, queue: cl.CommandQueue, additional_args: List[Any] = []) -> float:
    """
    Calculate a metric using an OpenCL kernel.

    Parameters:
        program (cl.Program): The OpenCL program containing the kernel.
        kernel_name (str): The name of the kernel to execute.
        params (np.ndarray): The parameters to pass to the kernel.
        alpha_values (np.ndarray): The alpha values for scaling.
        context (cl.Context): The OpenCL context.
        queue (cl.CommandQueue): The OpenCL command queue.
        additional_args (List[Any], optional): Additional arguments to pass to the kernel.

    Returns:
        float: The calculated metric.

    Raises:
        Exception: If an error occurs during the calculation.
    """
    try:
        # Create buffers for parameters and alpha values
        param_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=params)
        alpha_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=alpha_values)
        result_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, size=params.nbytes)

        # Prepare arguments for the kernel
        args = [param_buf, alpha_buf, result_buf] + additional_args

        # Execute the kernel
        result = execute_kernel(program, kernel_name, args, context, queue)

        # Sum the results to get the final metric
        metric = np.sum(result)
        logging.debug(f"calculate_metric: Calculated metric for kernel {kernel_name}: {metric}")
        return metric
    except Exception as e:
        logging.error(f"calculate_metric: Error calculating metric for kernel {kernel_name}: {e}")
        raise


def compute_intelligence(combination: Dict[str, float], opencl_manager: OpenCLManager) -> Optional[float]:
    """
    Compute the intelligence metric for a given combination of parameters using OpenCL for parallel computation.

    Parameters:
        combination (Dict[str, float]): A dictionary containing the parameter values for the combination.
        opencl_manager (OpenCLManager): An instance of OpenCLManager to manage OpenCL resources.

    Returns:
        Optional[float]: The computed intelligence metric, or None if an error occurs.
    """
    logging.debug(f"Starting computation of intelligence metric with combination: {combination}")

    try:
        # Calculate entropy (H_X)
        H_X = calculate_metric(
            opencl_manager.programs['entropy'], 'calculate_entropy',
            np.array(combination['probabilities_set'], dtype=np.float32),
            np.array([alpha_parameters["alpha_H"], alpha_parameters["alpha_Pi"], alpha_parameters["alpha_log"]], dtype=np.float32),
            opencl_manager.context, opencl_manager.queue,
            [np.int32(len(combination['probabilities_set']))]
        )
        
        # Calculate mutual information (I_XY)
        I_XY = calculate_metric(
            opencl_manager.programs['mutual_information'], 'calculate_mutual_information',
            np.array([combination['H_X_set'], combination['H_Y_set'], combination['H_XY_set']], dtype=np.float32),
            np.array([alpha_parameters["alpha_I"], alpha_parameters["alpha_HX"], alpha_parameters["alpha_HY"], alpha_parameters["alpha_HXY"]], dtype=np.float32),
            opencl_manager.context, opencl_manager.queue
        )
        
        # Calculate operational efficiency (O)
        O = calculate_metric(
            opencl_manager.programs['operational'], 'calculate_operational_efficiency',
            np.array([combination['P_set'], combination['E_set']], dtype=np.float32),
            np.array([alpha_parameters["alpha_O"], alpha_parameters["alpha_P"], alpha_parameters["alpha_E"]], dtype=np.float32),
            opencl_manager.context, opencl_manager.queue
        )
        
        # Calculate error management (Em)
        Em = calculate_metric(
            opencl_manager.programs['error'], 'calculate_error_management',
            np.array([combination['error_detection_rate_set'], combination['correction_capability_set']], dtype=np.float32),
            np.array([alpha_parameters["alpha_Em"], alpha_parameters["alpha_Error_Detection"], alpha_parameters["alpha_Correction"]], dtype=np.float32),
            opencl_manager.context, opencl_manager.queue
        )
        
        # Calculate adaptability (A)
        A = calculate_metric(
            opencl_manager.programs['adaptability'], 'calculate_adaptability',
            np.array(combination['adaptation_rate_set'], dtype=np.float32),
            np.array([alpha_parameters["alpha_A"], alpha_parameters["alpha_Adaptation_Rate"]], dtype=np.float32),
            opencl_manager.context, opencl_manager.queue
        )
        
        # Calculate volume (Volume)
        Volume = calculate_metric(
            opencl_manager.programs['volume'], 'calculate_volume',
            np.array(combination['spatial_scale_set'], dtype=np.float32),
            np.array([alpha_parameters["alpha_Volume"], alpha_parameters["alpha_Spatial_Scale"]], dtype=np.float32),
            opencl_manager.context, opencl_manager.queue
        )
        
        # Calculate time (Time)
        Time = calculate_metric(
            opencl_manager.programs['time'], 'calculate_time',
            np.array(combination['temporal_scale_set'], dtype=np.float32),
            np.array([alpha_parameters["alpha_t"], alpha_parameters["alpha_Temporal_Scale"]], dtype=np.float32),
            opencl_manager.context, opencl_manager.queue
        )

        # Validate Volume and Time to ensure they are non-zero and non-negligible
        if np.isclose(Volume, 0) or np.isclose(Time, 0):
            raise ValueError("Volume and Time must be non-zero and non-negligible.")

        # Compute the intelligence metric
        intelligence_metric = alpha_parameters["k"] * (H_X * I_XY * O * Em * A) / (Volume * Time)
        logging.debug(f"Computed intelligence metric: {intelligence_metric}")
        return intelligence_metric

    except Exception as e:
        logging.error(f"Failed to compute intelligence for combination {combination}: {e}")
        return None


# Plotting Functions
def plot_heatmap(data: pd.DataFrame) -> None:
    logging.debug(f"Starting to plot heatmap for data with shape {data.shape} and columns {data.columns.tolist()}")
    try:
        plt.figure(figsize=(14, 10))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Matrix of Parameters and Intelligence')
        plt.savefig('correlation_matrix.png')
        plt.show()
    except Exception as e:
        logging.error(f"Failed to plot heatmap: {e}")

def plot_pairplot(data: pd.DataFrame) -> None:
    logging.debug(f"Starting to plot pairplot for data with shape {data.shape} and columns {data.columns.tolist()}")
    try:
        sns.pairplot(data, hue='Intelligence', palette='viridis')
        plt.savefig('pairplot.png')
        plt.show()
    except Exception as e:
        logging.error(f"Failed to plot pairplot: {e}")

def plot_jointplot(data: pd.DataFrame) -> None:
    logging.debug(f"Starting to plot jointplot for data with shape {data.shape} and columns {data.columns.tolist()}")
    try:
        sns.jointplot(data=data, x='H_X', y='Intelligence', kind='hex', cmap='Blues')
        plt.savefig('jointplot.png')
        plt.show()
    except Exception as e:
        logging.error(f"Failed to plot jointplot: {e}")

def plot_histograms(data: pd.DataFrame) -> None:
    logging.debug(f"Starting to plot histograms for data with shape {data.shape} and columns {data.columns.tolist()}")
    try:
        for column in data.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(data[column], bins=20, kde=True, color='magenta')
            plt.title(f'Distribution of {column}')
            plt.savefig(f'{column}_distribution.png')
            plt.show()
    except Exception as e:
        logging.error(f"Failed to plot histograms: {e}")

def plot_boxplots(data: pd.DataFrame) -> None:
    logging.debug(f"Starting to plot boxplots for data with shape {data.shape} and columns {data.columns.tolist()}")
    try:
        for column in data.columns[:-1]:
            plt.figure(figsize=(12, 8))
            sns.boxplot(x='Intelligence', y=column, data=data)
            plt.title(f'Intelligence vs {column}')
            plt.savefig(f'Intelligence_vs_{column}.png')
            plt.show()
    except Exception as e:
        logging.error(f"Failed to plot boxplots: {e}")

def plot_violinplot(data: pd.DataFrame) -> None:
    logging.debug(f"Starting to plot violinplot for data with shape {data.shape} and columns {data.columns.tolist()}")
    try:
        plt.figure(figsize=(12, 8))
        sns.violinplot(data=data, x='Error Detection Rate', y='Correction Capability', scale='width', inner='quartile')
        plt.title('Error Detection Rate vs Correction Capability')
        plt.savefig('Error_Detection_vs_Correction_Capability_violin.png')
        plt.show()
    except Exception as e:
        logging.error(f"Failed to plot violinplot: {e}")

def plot_scatter_matrix(data: pd.DataFrame) -> None:
    logging.debug(f"Starting to plot scatter matrix for data with shape {data.shape} and columns {data.columns.tolist()}")
    try:
        plt.figure(figsize=(20, 15))
        pd.plotting.scatter_matrix(data, alpha=0.8, figsize=(20, 15), diagonal='kde')
        plt.savefig('scatter_matrix.png')
        plt.show()
    except Exception as e:
        logging.error(f"Failed to plot scatter matrix: {e}")

def plot_ridge_plot(data: pd.DataFrame) -> None:
    logging.debug(f"Starting to plot ridge plot for data with shape {data.shape} and columns {data.columns.tolist()}")
    try:
        fig, axes = joypy.joyplot(data, by='Intelligence', figsize=(12, 8), colormap=plt.cm.viridis, alpha=0.8)
        plt.title('Ridge Plot of Parameters Grouped by Intelligence')
        plt.savefig('ridge_plot.png')
        plt.show()
    except Exception as e:
        logging.error(f"Failed to plot ridge plot: {e}")

def plot_3d_scatter(data: pd.DataFrame) -> None:
    logging.debug(f"Starting to plot 3D scatter plot for data with shape {data.shape} and columns {data.columns.tolist()}")
    try:
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data['H_X'], data['P'], data['Intelligence'], c='r', marker='o')
        ax.set_xlabel('H_X')
        ax.set_ylabel('P')
        ax.set_zlabel('Intelligence')
        plt.title('3D Scatter Plot of H_X, P and Intelligence')
        plt.savefig('3D_scatter_H_X_P_Intelligence.png')
        plt.show()
    except Exception as e:
        logging.error(f"Failed to plot 3D scatter: {e}")

def visualize_results(results: pd.DataFrame) -> None:
    logging.debug(f"Starting to visualize results with shape {results.shape} and columns {results.columns.tolist()}")
    try:
        numeric_results = results.drop(columns=['Probabilities'])
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

def main() -> None:
    logging.info("Starting Universal Intelligence Potential simulation.")
    try:
        opencl_manager = OpenCLManager()
        parameter_manager = ParameterManager()
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

