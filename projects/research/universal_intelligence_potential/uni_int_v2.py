import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Dict, Any
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

# Setup OpenCL environment
platform = cl.get_platforms()[0]  # Select the first platform
device = platform.get_devices()[0]  # Select the first device on this platform
context = cl.Context([device])  # Create a context with the selected device
queue = cl.CommandQueue(context)  # Create a command queue for the selected device

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

# OpenCL kernel for calculating entropy
entropy_kernel_code = """
__kernel void calculate_entropy(__global const float *probabilities, __global const float *alpha_params, __global float *results, const int size) {
    int i = get_global_id(0);
    if (i < size) {
        float alpha_H = alpha_params[0];
        float alpha_Pi = alpha_params[1];
        float alpha_log = alpha_params[2];
        float prob = probabilities[i];
        results[i] = alpha_H * (-prob * log(alpha_log * prob) * alpha_Pi);
    }
}
"""
entropy_program = cl.Program(context, entropy_kernel_code).build()

def calculate_entropy(probabilities: np.ndarray, alpha_params: Dict[str, float]) -> float:
    """
    Calculate the Shannon entropy of a probability distribution using scaling factors and OpenCL for parallel processing.
    
    Parameters:
        probabilities (np.ndarray): The probability distribution array.
        alpha_params (Dict[str, float]): Dictionary of alpha parameters for scaling.
    
    Returns:
        float: The calculated Shannon entropy.
    
    Raises:
        Exception: If an error occurs during the calculation.
    """
    try:
        alpha_values = np.array([alpha_params["alpha_H"], alpha_params["alpha_Pi"], alpha_params["alpha_log"]], dtype=np.float32)
        prob_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=probabilities)
        alpha_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=alpha_values)
        result_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, probabilities.nbytes)
        entropy_program.calculate_entropy(queue, probabilities.shape, None, prob_buf, alpha_buf, result_buf, np.int32(len(probabilities)))
        result = np.empty_like(probabilities)
        cl.enqueue_copy(queue, result, result_buf).wait()
        entropy = np.sum(result)
        logging.debug(f"calculate_entropy: Calculated Shannon entropy: {entropy}")
        progress_bar.update(1)
        return entropy
    except Exception as e:
        logging.error(f"calculate_entropy: Error calculating Shannon entropy: {e}")
        progress_bar.update(1)
        raise

# OpenCL kernel for calculating mutual information
mutual_information_kernel_code = """
__kernel void calculate_mutual_information(__global const float *H_X, __global const float *H_Y, __global const float *H_XY, __global const float *alpha_params, __global float *result) {
    int i = get_global_id(0);
    if (i < 1) {
        float alpha_I = alpha_params[0];
        float alpha_HX = alpha_params[1];
        float alpha_HY = alpha_params[2];
        float alpha_HXY = alpha_params[3];
        result[0] = alpha_I * (alpha_HX * H_X[0] + alpha_HY * H_Y[0] - alpha_HXY * H_XY[0]);
    }
}
"""
mutual_information_program = cl.Program(context, mutual_information_kernel_code).build()

def calculate_mutual_information(H_X: float, H_Y: float, H_XY: float, alpha_params: Dict[str, float]) -> float:
    """
    Calculate the mutual information based on entropies of X, Y, and their joint distribution using OpenCL for parallel processing.
    
    Parameters:
        H_X (float): Entropy of X.
        H_Y (float): Entropy of Y.
        H_XY (float): Joint entropy of X and Y.
        alpha_params (Dict[str, float]): Dictionary of alpha parameters for scaling.
    
    Returns:
        float: The calculated mutual information.
    
    Raises:
        Exception: If an error occurs during the calculation.
    """
    try:
        alpha_values = np.array([alpha_params["alpha_I"], alpha_params["alpha_HX"], alpha_params["alpha_HY"], alpha_params["alpha_HXY"]], dtype=np.float32)
        H_X_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.array([H_X], dtype=np.float32))
        H_Y_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.array([H_Y], dtype=np.float32))
        H_XY_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.array([H_XY], dtype=np.float32))
        alpha_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=alpha_values)
        result_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, np.float32(0).nbytes)
        mutual_information_program.calculate_mutual_information(queue, (1,), None, H_X_buf, H_Y_buf, H_XY_buf, alpha_buf, result_buf)
        result = np.empty(1, dtype=np.float32)
        cl.enqueue_copy(queue, result, result_buf).wait()
        mutual_info = result[0]
        logging.debug(f"calculate_mutual_information: Calculated mutual information: {mutual_info}")
        progress_bar.update(1)
        return mutual_info
    except Exception as e:
        logging.error(f"calculate_mutual_information: Error calculating mutual information: {e}")
        progress_bar.update(1)
        raise

# OpenCL kernel for calculating operational efficiency
operational_efficiency_kernel_code = """
__kernel void calculate_operational_efficiency(__global const float *P, __global const float *E, __global const float *alpha_params, __global float *result) {
    int i = get_global_id(0);
    if (i < 1) {
        float alpha_O = alpha_params[0];
        float alpha_P = alpha_params[1];
        float alpha_E = alpha_params[2];
        if (E[0] == 0) {
            result[0] = 0.0;  // Avoid division by zero
        } else {
            result[0] = alpha_O * (alpha_P * P[0] / (alpha_E * E[0]));
        }
    }
}
"""
operational_efficiency_program = cl.Program(context, operational_efficiency_kernel_code).build()

def calculate_operational_efficiency(P: float, E: float, alpha_params: Dict[str, float]) -> float:
    """
    Calculate the operational efficiency based on performance and energy consumption using OpenCL for parallel processing.
    
    Parameters:
        P (float): Performance measure, typically computational power or output rate.
        E (float): Energy consumption measure.
        alpha_params (Dict[str, float]): Dictionary of alpha parameters for scaling.
        
    Returns:
        float: The calculated operational efficiency.
        
    Raises:
        Exception: If an error occurs during the calculation.
    """
    try:
        alpha_values = np.array([alpha_params["alpha_O"], alpha_params["alpha_P"], alpha_params["alpha_E"]], dtype=np.float32)
        P_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.array([P], dtype=np.float32))
        E_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.array([E], dtype=np.float32))
        alpha_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=alpha_values)
        result_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, np.float32(0).nbytes)
        operational_efficiency_program.calculate_operational_efficiency(queue, (1,), None, P_buf, E_buf, alpha_buf, result_buf)
        result = np.empty(1, dtype=np.float32)
        cl.enqueue_copy(queue, result, result_buf).wait()
        efficiency = result[0]
        logging.debug(f"calculate_operational_efficiency: Calculated operational efficiency: {efficiency}")
        progress_bar.update(1)
        return efficiency
    except Exception as e:
        logging.error(f"calculate_operational_efficiency: Error calculating operational efficiency: {e}")
        progress_bar.update(1)
        raise

# OpenCL kernel for calculating error management
error_management_kernel_code = """
__kernel void calculate_error_management(__global const float *error_detection_rate, __global const float *correction_capability, __global const float *alpha_params, __global float *result) {
    int i = get_global_id(0);
    if (i < 1) {
        float alpha_Em = alpha_params[0];
        float alpha_Error_Detection = alpha_params[1];
        float alpha_Correction = alpha_params[2];
        result[0] = alpha_Em * (alpha_Error_Detection * error_detection_rate[0] * alpha_Correction * correction_capability[0]);
    }
}
"""
error_management_program = cl.Program(context, error_management_kernel_code).build()

def calculate_error_management(error_detection_rate: float, correction_capability: float, alpha_params: Dict[str, float]) -> float:
    """
    Calculate the error management effectiveness based on error detection and correction capabilities using OpenCL for parallel processing.
    
    Parameters:
        error_detection_rate (float): The rate at which errors are detected.
        correction_capability (float): The capability of the system to correct detected errors.
        alpha_params (Dict[str, float]): Dictionary of alpha parameters for scaling.
        
    Returns:
        float: The calculated error management effectiveness.
        
    Raises:
        Exception: If an error occurs during the calculation.
    """
    try:
        alpha_values = np.array([alpha_params["alpha_Em"], alpha_params["alpha_Error_Detection"], alpha_params["alpha_Correction"]], dtype=np.float32)
        error_detection_rate_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.array([error_detection_rate], dtype=np.float32))
        correction_capability_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.array([correction_capability], dtype=np.float32))
        alpha_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=alpha_values)
        result_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, np.float32(0).nbytes)
        error_management_program.calculate_error_management(queue, (1,), None, error_detection_rate_buf, correction_capability_buf, alpha_buf, result_buf)
        result = np.empty(1, dtype=np.float32)
        cl.enqueue_copy(queue, result, result_buf).wait()
        error_management_value = result[0]
        logging.debug(f"calculate_error_management: Calculated error management effectiveness: {error_management_value}")
        progress_bar.update(1)
        return error_management_value
    except Exception as e:
        logging.error(f"calculate_error_management: Error calculating error management effectiveness: {e}")
        progress_bar.update(1)
        raise

# OpenCL kernel for calculating adaptability
adaptability_kernel_code = """
__kernel void calculate_adaptability(__global const float *adaptation_rate, __global const float *alpha_params, __global float *result) {
    int i = get_global_id(0);
    if (i < 1) {
        float alpha_A = alpha_params[0];
        float alpha_Adaptation_Rate = alpha_params[1];
        result[0] = alpha_A * (alpha_Adaptation_Rate * adaptation_rate[0]);
    }
}
"""
adaptability_program = cl.Program(context, adaptability_kernel_code).build()

def calculate_adaptability(adaptation_rate: float, alpha_params: Dict[str, float]) -> float:
    """
    Calculate the adaptability based on the adaptation rate.
    
    Parameters:
        adaptation_rate (float): The rate at which the system can adapt to changes.
        alpha_params (Dict[str, float]): Dictionary of alpha parameters for scaling.
        
    Returns:
        float: The calculated adaptability.
        
    Raises:
        Exception: If an error occurs during the calculation.
    """
    try:
        alpha_values = np.array([alpha_params["alpha_A"], alpha_params["alpha_Adaptation_Rate"]], dtype=np.float32)
        adaptation_rate_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.array([adaptation_rate], dtype=np.float32))
        alpha_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=alpha_values)
        result_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, np.float32(0).nbytes)
        adaptability_program.calculate_adaptability(queue, (1,), None, adaptation_rate_buf, alpha_buf, result_buf)
        result = np.empty(1, dtype=np.float32)
        cl.enqueue_copy(queue, result, result_buf).wait()
        adaptability_value = result[0]
        logging.debug(f"calculate_adaptability: Calculated adaptability: {adaptability_value}")
        progress_bar.update(1)
        return adaptability_value
    except Exception as e:
        logging.error(f"calculate_adaptability: Error calculating adaptability: {e}")
        progress_bar.update(1)
        raise

# OpenCL kernel for calculating volume
volume_kernel_code = """
__kernel void calculate_volume(__global const float *spatial_scale, __global const float *alpha_params, __global float *result) {
    int i = get_global_id(0);
    if (i < 1) {
        float alpha_Volume = alpha_params[0];
        float alpha_Spatial_Scale = alpha_params[1];
        result[0] = alpha_Volume * (alpha_Spatial_Scale * spatial_scale[0]);
    }
}
"""
volume_program = cl.Program(context, volume_kernel_code).build()

def calculate_volume(spatial_scale: float, alpha_params: Dict[str, float]) -> float:
    """
    Calculate the volume based on the spatial scale.
    
    Parameters:
        spatial_scale (float): The spatial scale factor, typically representing physical dimensions.
        alpha_params (Dict[str, float]): Dictionary of alpha parameters for scaling.
        
    Returns:
        float: The calculated volume.
        
    Raises:
        Exception: If an error occurs during the calculation.
    """
    try:
        alpha_values = np.array([alpha_params["alpha_Volume"], alpha_params["alpha_Spatial_Scale"]], dtype=np.float32)
        spatial_scale_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.array([spatial_scale], dtype=np.float32))
        alpha_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=alpha_values)
        result_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, np.float32(0).nbytes)
        volume_program.calculate_volume(queue, (1,), None, spatial_scale_buf, alpha_buf, result_buf)
        result = np.empty(1, dtype=np.float32)
        cl.enqueue_copy(queue, result, result_buf).wait()
        volume_value = result[0]
        logging.debug(f"calculate_volume: Calculated volume: {volume_value}")
        progress_bar.update(1)
        return volume_value
    except Exception as e:
        logging.error(f"calculate_volume: Error calculating volume: {e}")
        progress_bar.update(1)
        raise

# OpenCL kernel for calculating time
time_kernel_code = """
__kernel void calculate_time(__global const float *temporal_scale, __global const float *alpha_params, __global float *result) {
    int i = get_global_id(0);
    if (i < 1) {
        float alpha_t = alpha_params[0];
        float alpha_Temporal_Scale = alpha_params[1];
        result[0] = alpha_t * (alpha_Temporal_Scale * temporal_scale[0]);
    }
}
"""
time_program = cl.Program(context, time_kernel_code).build()

def calculate_time(temporal_scale: float, alpha_params: Dict[str, float]) -> float:
    """
    Calculate the time based on the temporal scale.
    
    Parameters:
        temporal_scale (float): The temporal scale factor, typically representing time dimensions.
        alpha_params (Dict[str, float]): Dictionary of alpha parameters for scaling.
        
    Returns:
        float: The calculated time.
        
    Raises:
        Exception: If an error occurs during the calculation.
    """
    try:
        alpha_values = np.array([alpha_params["alpha_t"], alpha_params["alpha_Temporal_Scale"]], dtype=np.float32)
        temporal_scale_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.array([temporal_scale], dtype=np.float32))
        alpha_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=alpha_values)
        result_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, np.float32(0).nbytes)
        time_program.calculate_time(queue, (1,), None, temporal_scale_buf, alpha_buf, result_buf)
        result = np.empty(1, dtype=np.float32)
        cl.enqueue_copy(queue, result, result_buf).wait()
        time_value = result[0]
        logging.debug(f"calculate_time: Calculated time: {time_value}")
        progress_bar.update(1)
        return time_value
    except Exception as e:
        logging.error(f"calculate_time: Error calculating time: {e}")
        progress_bar.update(1)
        raise

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



def main():
    """
    Main function to execute the model simulation, output results, and visualize the data with enhanced precision and detail using OpenCL for parallel computation.
    """
    # Check if parameter ranges file exists and load it if available
    if os.path.exists('parameter_ranges.npz'):
        with np.load('parameter_ranges.npz') as data:
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
    else:
        # Define a refined range of values for each component to simulate various scenarios with optimized memory usage
        probabilities_set = [
            np.array([p1, p2, 1 - p1 - p2]) 
            for p1 in np.linspace(0.1, 0.9, 9) 
            for p2 in np.linspace(0.1, 0.9, 9) 
            if p1 + p2 <= 1
        ]
        
        # Define entropy and other parameters with refined granularity
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
        np.savez('parameter_ranges.npz', probabilities_set=probabilities_set, H_X_set=H_X_set, H_Y_set=H_Y_set, H_XY_set=H_XY_set,
                 P_set=P_set, E_set=E_set, error_detection_rate_set=error_detection_rate_set,
                 correction_capability_set=correction_capability_set, adaptation_rate_set=adaptation_rate_set,
                 spatial_scale_set=spatial_scale_set, temporal_scale_set=temporal_scale_set)

    # Prepare a DataFrame with enhanced structure to store the results
    results = pd.DataFrame(columns=[
        'Probabilities', 'H_X', 'H_Y', 'H_XY', 'P', 'E', 'Error Detection Rate',
        'Correction Capability', 'Adaptation Rate', 'Spatial Scale', 'Temporal Scale', 'Intelligence'
    ])

    # Generate all combinations of input parameters with improved efficiency
    all_combinations = list(itertools.product(
        probabilities_set, H_X_set, H_Y_set, H_XY_set, P_set, E_set,
        error_detection_rate_set, correction_capability_set, adaptation_rate_set,
        spatial_scale_set, temporal_scale_set
    ))

    # Process each combination with enhanced computational methods
    for idx, combination in enumerate(all_combinations):
        probabilities, H_X, H_Y, H_XY, P, E, error_detection_rate, correction_capability, adaptation_rate, spatial_scale, temporal_scale = combination

        # Calculate each component with refined algorithms
        H_X_value = calculate_entropy(probabilities, alpha_parameters)
        I_XY_value = calculate_mutual_information(H_X, H_Y, H_XY, alpha_parameters)
        O_value = calculate_operational_efficiency(P, E, alpha_parameters)
        Em_value = calculate_error_management(error_detection_rate, correction_capability, alpha_parameters)
        A_value = calculate_adaptability(adaptation_rate, alpha_parameters)
        Volume_value = calculate_volume(spatial_scale, alpha_parameters)
        Time_value = calculate_time(temporal_scale, alpha_parameters)

        # Compute intelligence with an optimized formula
        I = alpha_parameters["k"] * (H_X_value * I_XY_value * O_value * Em_value * A_value) / (Volume_value * Time_value)

        # Append results to the DataFrame with enhanced data handling
        results = pd.concat([results, pd.DataFrame({
            'Probabilities': [probabilities], 'H_X': [H_X], 'H_Y': [H_Y], 'H_XY': [H_XY],
            'P': [P], 'E': [E], 'Error Detection Rate': [error_detection_rate],
            'Correction Capability': [correction_capability], 'Adaptation Rate': [adaptation_rate],
            'Spatial Scale': [spatial_scale], 'Temporal Scale': [temporal_scale], 'Intelligence': [I]
        })], ignore_index=True)

        # Update progress bar
        progress_bar.update(1)
        sys.stdout.write(f"\rmain: Progress: {idx+1}/{total_operations} | Last operation: Calculated Intelligence: {I:.2f}")

    # Close progress bar after all operations
    progress_bar.close()

    # Ensure all columns are numeric for correlation calculation
    numeric_results = results.drop(columns=['Probabilities'])

    # Output the results as a formatted table
    print(results.to_string(index=False))
    visualize_results(results)

if __name__ == "__main__":
    main()