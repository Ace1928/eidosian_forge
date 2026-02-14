import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Dict, Any, Optional
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joypy  # Library for ridge plots
from tqdm import tqdm
import pyopencl as cl
from mpl_toolkits.mplot3d import Axes3D

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

    def generate_probabilities_set(self) -> Dict[str, np.ndarray]:
        probabilities_set = {}
        for p1 in np.linspace(0.1, 0.9, 9):
            for p2 in np.linspace(0.1, 0.9, 9):
                if p1 + p2 <= 1:
                    key = f'prob_{p1}_{p2}'
                    value = np.array([p1, p2, 1 - p1 - p2])
                    probabilities_set[key] = value
                    logging.debug(f"Generated probability set {key}: {value}")
        return probabilities_set

    def load_parameter_ranges(self) -> Dict[str, np.ndarray]:
        return {
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

    def generate_combinations(self) -> List[Dict[str, Any]]:
        keys = list(self.parameter_ranges.keys())
        values = [self.parameter_ranges[key] for key in keys]
        combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
        return combinations

def execute_kernel(program: cl.Program, kernel_name: str, args: List, context: cl.Context, queue: cl.CommandQueue, global_size: Tuple[int], local_size: Tuple[int]) -> np.ndarray:
    try:
        kernel = getattr(program, kernel_name)
        kernel.set_args(*args)
        result_buf = args[-1]
        kernel(queue, global_size, local_size)
        result = np.empty(result_buf.size // np.dtype(np.float32).itemsize, dtype=np.float32)
        cl.enqueue_copy(queue, result, result_buf).wait()
        return result
    except Exception as e:
        logging.error(f"Error executing kernel {kernel_name}: {e}")
        raise

def calculate_metric(program: cl.Program, kernel_name: str, params: np.ndarray, alpha_values: np.ndarray, context: cl.Context, queue: cl.CommandQueue, global_size: Tuple[int], local_size: Tuple[int], additional_args: List[Any] = []) -> float:
    try:
        param_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=params)
        alpha_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=alpha_values)
        result_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, params.nbytes)

        args = [param_buf, alpha_buf, result_buf] + additional_args
        kernel = getattr(program, kernel_name)
        kernel.set_args(*args)
        cl.enqueue_nd_range_kernel(queue, kernel, global_size, local_size)
        result = np.empty_like(params)
        cl.enqueue_copy(queue, result, result_buf).wait()

        metric = np.sum(result)
        logging.debug(f"calculate_metric: Calculated metric for kernel {kernel_name}: {metric}")
        return metric
    except Exception as e:
        logging.error(f"calculate_metric: Error calculating metric for kernel {kernel_name}: {e}")
        raise

def compute_intelligence(combination: Dict[str, Any], opencl_manager: OpenCLManager) -> Optional[float]:
    try:
        cumulative_intelligence_metric = 0.0
        count = 0
        global_size = (256,)
        local_size = (64,)

        for prob_key, prob_values in combination['probabilities_set'].items():
            logging.debug(f"Processing probability set for key: {prob_key} with values: {prob_values}")

            if isinstance(prob_values, list) and isinstance(prob_values[0], str):
                prob_values = prob_values[1:]
                logging.debug(f"Removed string label from prob_values: {prob_values}")

            prob_values_array: np.ndarray = np.array(prob_values, dtype=np.float32)
            logging.debug(f"Converted prob_values to numpy array: {prob_values_array}")

            H_X: float = calculate_metric(
                opencl_manager.load_all_programs()['entropy'], 'calculate_entropy',
                prob_values_array,
                np.array([alpha_parameters["alpha_H"], alpha_parameters["alpha_Pi"], alpha_parameters["alpha_log"]], dtype=np.float32),
                opencl_manager.context, opencl_manager.queue, global_size, local_size,
                [np.int32(len(prob_values))]
            )
            logging.debug(f"Calculated entropy H_X: {H_X} for prob_key: {prob_key}")

            H_X_set_array: np.ndarray = np.array(combination['H_X_set'], dtype=np.float32)
            H_Y_set_array: np.ndarray = np.array(combination['H_Y_set'], dtype=np.float32)
            H_XY_set_array: np.ndarray = np.array(combination['H_XY_set'], dtype=np.float32)
            logging.debug(f"Converted H_X_set, H_Y_set, H_XY_set to numpy arrays: {H_X_set_array}, {H_Y_set_array}, {H_XY_set_array}")

            I_XY: float = calculate_metric(
                opencl_manager.load_all_programs()['mutual_information'], 'calculate_mutual_information',
                np.array([H_X_set_array, H_Y_set_array, H_XY_set_array], dtype=np.float32),
                np.array([alpha_parameters["alpha_I"], alpha_parameters["alpha_HX"], alpha_parameters["alpha_HY"], alpha_parameters["alpha_HXY"]], dtype=np.float32),
                opencl_manager.context, opencl_manager.queue, global_size, local_size
            )
            logging.debug(f"Calculated mutual information I_XY: {I_XY} for prob_key: {prob_key}")

            P_set_array: np.ndarray = np.array(combination['P_set'], dtype=np.float32)
            E_set_array: np.ndarray = np.array(combination['E_set'], dtype=np.float32)
            logging.debug(f"Converted P_set and E_set to numpy arrays: {P_set_array}, {E_set_array}")

            O: float = calculate_metric(
                opencl_manager.load_all_programs()['operational_efficiency'], 'calculate_operational_efficiency',
                np.array([P_set_array, E_set_array], dtype=np.float32),
                np.array([alpha_parameters["alpha_O"], alpha_parameters["alpha_P"], alpha_parameters["alpha_E"]], dtype=np.float32),
                opencl_manager.context, opencl_manager.queue, global_size, local_size
            )
            logging.debug(f"Calculated operational efficiency O: {O} for prob_key: {prob_key}")

            error_detection_rate_set_array: np.ndarray = np.array(combination['error_detection_rate_set'], dtype=np.float32)
            correction_capability_set_array: np.ndarray = np.array(combination['correction_capability_set'], dtype=np.float32)
            logging.debug(f"Converted error_detection_rate_set and correction_capability_set to numpy arrays: {error_detection_rate_set_array}, {correction_capability_set_array}")

            Em: float = calculate_metric(
                opencl_manager.load_all_programs()['error_management'], 'calculate_error_management',
                np.array([error_detection_rate_set_array, correction_capability_set_array], dtype=np.float32),
                np.array([alpha_parameters["alpha_Em"], alpha_parameters["alpha_Error_Detection"], alpha_parameters["alpha_Correction"]], dtype=np.float32),
                opencl_manager.context, opencl_manager.queue, global_size, local_size
            )
            logging.debug(f"Calculated error management Em: {Em} for prob_key: {prob_key}")

            adaptation_rate_set_array: np.ndarray = np.array(combination['adaptation_rate_set'], dtype=np.float32)
            logging.debug(f"Converted adaptation_rate_set to numpy array: {adaptation_rate_set_array}")

            A: float = calculate_metric(
                opencl_manager.load_all_programs()['adaptability'], 'calculate_adaptability',
                adaptation_rate_set_array,
                np.array([alpha_parameters["alpha_A"], alpha_parameters["alpha_Adaptation_Rate"]], dtype=np.float32),
                opencl_manager.context, opencl_manager.queue, global_size, local_size
            )
            logging.debug(f"Calculated adaptability A: {A} for prob_key: {prob_key}")

            spatial_scale_set_array: np.ndarray = np.array(combination['spatial_scale_set'], dtype=np.float32)
            temporal_scale_set_array: np.ndarray = np.array(combination['temporal_scale_set'], dtype=np.float32)
            logging.debug(f"Converted spatial_scale_set and temporal_scale_set to numpy arrays: {spatial_scale_set_array}, {temporal_scale_set_array}")

            Volume: float = calculate_metric(
                opencl_manager.load_all_programs()['volume'], 'calculate_volume',
                spatial_scale_set_array,
                np.array([alpha_parameters["alpha_Volume"], alpha_parameters["alpha_Spatial_Scale"]], dtype=np.float32),
                opencl_manager.context, opencl_manager.queue, global_size, local_size
            )
            logging.debug(f"Calculated volume: {Volume} for prob_key: {prob_key}")

            Time: float = calculate_metric(
                opencl_manager.load_all_programs()['time'], 'calculate_time',
                temporal_scale_set_array,
                np.array([alpha_parameters["alpha_t"], alpha_parameters["alpha_Temporal_Scale"]], dtype=np.float32),
                opencl_manager.context, opencl_manager.queue, global_size, local_size
            )
            logging.debug(f"Calculated time: {Time} for prob_key: {prob_key}")

            if np.isclose(Volume, 0) or np.isclose(Time, 0):
                error_message: str = f"Volume and Time must be non-zero and non-negligible for {prob_key}."
                logging.error(error_message)
                raise ValueError(error_message)

            intelligence_metric: float = (H_X * I_XY * O * Em * A) / (Volume * Time)
            logging.debug(f"Calculated intelligence metric: {intelligence_metric} for prob_key: {prob_key}")

            cumulative_intelligence_metric += intelligence_metric
            count += 1
            logging.debug(f"Updated cumulative intelligence metric: {cumulative_intelligence_metric}, count: {count}")

        average_intelligence_metric: Optional[float] = cumulative_intelligence_metric / count if count > 0 else None
        logging.debug(f"Final average intelligence metric: {average_intelligence_metric}")
        return average_intelligence_metric
    except Exception as e:
        logging.error(f"Failed to compute intelligence for combination {combination}: {e}")
        return None

class DataVisualizer:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        logging.debug(f"DataVisualizer initialized with data of shape {data.shape} and columns {data.columns.tolist()}")

    def plot_heatmap(self) -> None:
        logging.debug(f"Starting to plot heatmap.")
        try:
            plt.figure(figsize=(14, 10))
            sns.heatmap(self.data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
            plt.title('Correlation Matrix of Parameters and Intelligence')
            plt.savefig('correlation_matrix.png')
            plt.show()
        except Exception as e:
            logging.error(f"Failed to plot heatmap: {e}")

    def plot_pairplot(self, hue: str = 'Intelligence') -> None:
        logging.debug(f"Starting to plot pairplot with hue {hue}.")
        try:
            sns.pairplot(self.data, hue=hue, palette='viridis')
            plt.savefig('pairplot.png')
            plt.show()
        except Exception as e:
            logging.error(f"Failed to plot pairplot: {e}")

    def plot_jointplot(self, x: str = 'H_X', y: str = 'Intelligence') -> None:
        logging.debug(f"Starting to plot jointplot with x={x} and y={y}.")
        try:
            sns.jointplot(data=self.data, x=x, y=y, kind='hex', cmap='Blues')
            plt.savefig('jointplot.png')
            plt.show()
        except Exception as e:
            logging.error(f"Failed to plot jointplot: {e}")

    def plot_histograms(self) -> None:
        logging.debug(f"Starting to plot histograms.")
        try:
            for column in self.data.columns:
                plt.figure(figsize=(10, 6))
                sns.histplot(self.data[column], bins=20, kde=True, color='magenta')
                plt.title(f'Distribution of {column}')
                plt.savefig(f'{column}_distribution.png')
                plt.show()
        except Exception as e:
            logging.error(f"Failed to plot histograms: {e}")

    def plot_boxplots(self, x: str = 'Intelligence') -> None:
        logging.debug(f"Starting to plot boxplots with x={x}.")
        try:
            for column in self.data.columns:
                if column != x:
                    plt.figure(figsize=(12, 8))
                    sns.boxplot(x=x, y=column, data=self.data)
                    plt.title(f'{x} vs {column}')
                    plt.savefig(f'{x}_vs_{column}.png')
                    plt.show()
        except Exception as e:
            logging.error(f"Failed to plot boxplots: {e}")

    def plot_violinplot(self, x: str, y: str) -> None:
        logging.debug(f"Starting to plot violinplot with x={x} and y={y}.")
        try:
            plt.figure(figsize=(12, 8))
            sns.violinplot(data=self.data, x=x, y=y, scale='width', inner='quartile')
            plt.title(f'{x} vs {y}')
            plt.savefig(f'{x}_vs_{y}_violin.png')
            plt.show()
        except Exception as e:
            logging.error(f"Failed to plot violinplot: {e}")

    def plot_scatter_matrix(self) -> None:
        logging.debug(f"Starting to plot scatter matrix.")
        try:
            plt.figure(figsize=(20, 15))
            pd.plotting.scatter_matrix(self.data, alpha=0.8, figsize=(20, 15), diagonal='kde')
            plt.savefig('scatter_matrix.png')
            plt.show()
        except Exception as e:
            logging.error(f"Failed to plot scatter matrix: {e}")

    def plot_ridge_plot(self, by: str = 'Intelligence') -> None:
        logging.debug(f"Starting to plot ridge plot grouped by {by}.")
        try:
            fig, axes = joypy.joyplot(self.data, by=by, figsize=(12, 8), colormap=plt.cm.viridis, alpha=0.8)
            plt.title('Ridge Plot of Parameters Grouped by Intelligence')
            plt.savefig('ridge_plot.png')
            plt.show()
        except Exception as e:
            logging.error(f"Failed to plot ridge plot: {e}")

    def plot_3d_scatter(self, x: str = 'H_X', y: str = 'P', z: str = 'Intelligence') -> None:
        logging.debug(f"Starting to plot 3D scatter plot with x={x}, y={y}, and z={z}.")
        try:
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.data[x], self.data[y], self.data[z], c='r', marker='o')
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.set_zlabel(z)
            plt.title(f'3D Scatter Plot of {x}, {y} and {z}')
            plt.savefig('3D_scatter_plot.png')
            plt.show()
        except Exception as e:
            logging.error(f"Failed to plot 3D scatter: {e}")

    def visualize_results(self) -> None:
        logging.debug(f"Starting to visualize results.")
        try:
            numeric_results = self.data.select_dtypes(include=[np.number])
            self.plot_heatmap()
            self.plot_pairplot()
            self.plot_jointplot()
            self.plot_histograms()
            self.plot_boxplots()
            self.plot_violinplot('Error Detection Rate', 'Correction Capability')
            self.plot_scatter_matrix()
            self.plot_ridge_plot()
            self.plot_3d_scatter()
        except Exception as e:
            logging.error(f"Failed to visualize results: {e}")
        finally:
            logging.info("Visualization process completed.")

# Example usage:
# Assuming 'results_df' is a DataFrame obtained from the main function
# visualizer = DataVisualizer(results_df)
# visualizer.visualize_results()


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

        # Validate the generated probability set
        probabilities_set = parameter_manager.generate_probabilities_set()
        print("Generated probabilities_set:")
        for key, value in probabilities_set.items():
            print(f"{key}: {value}")

        # Iterate over all combinations
        for combination in tqdm(combinations, desc="Processing Combinations", unit="combination"):
            # Compute intelligence metric for the combination
            intelligence_metric = compute_intelligence(combination, opencl_manager)
            if intelligence_metric is not None:
                results.append({**combination, "Intelligence": intelligence_metric})

        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        results_df.to_csv("simulation_results.csv", index=False)

        # Visualize results
        visualizer = DataVisualizer(results_df)
        visualizer.visualize_results()

    except Exception as e:
        logging.error(f"An error occurred during the simulation: {e}")
        raise
    finally:
        logging.info("Simulation completed.")

if __name__ == "__main__":
    main()