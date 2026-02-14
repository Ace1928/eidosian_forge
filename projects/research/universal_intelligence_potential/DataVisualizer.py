import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
import joypy
from bokeh.plotting import figure, show, save
from bokeh.io import output_file
from bokeh.resources import INLINE
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Union, List, Tuple, Any
from datetime import datetime
import os
import logging
import itertools

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class DataVisualizer:
    def __init__(self, data: Optional[Union[pd.DataFrame, str]] = None) -> None:
        self.data: Optional[pd.DataFrame] = self._load_data(data) if data is not None else None
        if self.data is not None:
            logging.debug(f"DataVisualizer initialized with data of shape {self.data.shape} and columns {self.data.columns.tolist()}")

    def _load_data(self, data: Union[pd.DataFrame, str]) -> pd.DataFrame:
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
        
    def clean_data(self):
        logging.debug("Cleaning data by replacing infs and NaNs.")
        self.data.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.data.dropna(inplace=True)
        logging.debug("Data cleaned successfully.")

    def _validate_columns(self, columns: List[str], data: Optional[pd.DataFrame] = None) -> None:
        data = self._load_data(data) if data is not None else self.data
        if data is None:
            raise ValueError("No data provided for column validation.")
        missing_columns = [col for col in columns if col not in data.columns]
        if missing_columns:
            error_message = f"The following columns are missing in the dataset: {missing_columns}"
            logging.warning(error_message)
            raise ValueError(error_message)

    def _generate_filename(self, base_name: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base_name}_{timestamp}.png"

    def _save_and_show_plot(self, plot_type: str, filename: str, plot_object: Any = None, plot_library: str = 'matplotlib', **kwargs) -> None:
        directory: str = os.path.join('/home/lloyd/UniversalIntelligencePotential/DataVisualizerOutput/', plot_type)
        os.makedirs(directory, exist_ok=True)
        filepath: str = os.path.join(directory, filename)
        
        plot_libraries = ['matplotlib', 'seaborn', 'plotly', 'joypy', 'bokeh']
        if plot_library not in plot_libraries:
            raise ValueError(f"Unsupported plotting library: {plot_library}")
        
        try:
            if plot_library == 'matplotlib':
                if plot_object is None:
                    plot_object = plt
                plot_object.gcf().set_size_inches(20, 15)
                plot_object.savefig(filepath, bbox_inches='tight', pad_inches=0.1, **kwargs)
                plot_object.show()
                plt.pause(0.001)
                plot_object.close()
            elif plot_library == 'seaborn':
                if plot_object is None:
                    raise ValueError("plot_object must be provided when using seaborn.")
                plot_object.figure.set_size_inches(20, 15)
                plot_object.figure.savefig(filepath, bbox_inches='tight', pad_inches=0.1, **kwargs)
                plot_object.figure.show()
                plt.pause(0.001)
                plt.close(plot_object.figure)
            elif plot_library == 'plotly':
                if plot_object is None:
                    raise ValueError("plot_object must be provided when using plotly.")
                plot_object.update_layout(width=1920, height=1080)
                plot_object.write_image(filepath, **kwargs)
                plot_object.show()
            elif plot_library == 'joypy':
                if plot_object is None:
                    raise ValueError("plot_object must be provided when using joypy.")
                fig, axes = plot_object
                fig.set_size_inches(20, 15)
                fig.savefig(filepath, bbox_inches='tight', pad_inches=0.1, **kwargs)
                plt.show()
                plt.pause(0.001)
                plt.close(fig)
            elif plot_library == 'bokeh':
                if plot_object is None:
                    raise ValueError("plot_object must be provided when using bokeh.")
                plot_object.plot_width = 1920
                plot_object.plot_height = 1080
                save(plot_object, filename=filepath, resources=INLINE, **kwargs)
                show(plot_object)
            plt.close('all')
            logging.info(f"Plot saved as {filepath}")
        except Exception as e:
            logging.error(f"Failed to save and display plot using {plot_library}: {e}", exc_info=True)
            plt.close('all')
            raise RuntimeError(f"Failed to save and display plot using {plot_library}.") from e

    def plot_scatter_matrix(self, data: Optional[pd.DataFrame] = None, alpha: float = 0.8, figsize: Tuple[int, int] = (20, 15), diagonal: str = 'kde') -> None:
        logging.debug("Starting to plot scatter matrix.")
        
        data = self._load_data(data) if data is not None else self.data
        if data is None:
            raise ValueError("No data provided for plotting.")
        
        self.clean_data()
        
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            error_message = "No numeric columns available for scatter matrix."
            logging.warning(error_message)
            return
        
        try:
            numeric_data = numeric_data.dropna()
            scatter_matrix = sns.pairplot(
                numeric_data,
                diag_kind=diagonal,
                kind='scatter',
                height=figsize[1] / len(numeric_data.columns),
                aspect=figsize[0] / figsize[1],
                corner=True,
                plot_kws={'alpha': alpha, 'edgecolor': 'black', 'linewidth': 0.5},
                diag_kws={'fill': True}
            )
            scatter_matrix.fig.suptitle('Scatter Matrix of Parameters and Intelligence', fontsize=16)
            scatter_matrix.fig.subplots_adjust(top=0.95)
            self._save_and_show_plot('scatter_matrix', self._generate_filename('scatter_matrix'), plot_object=scatter_matrix, plot_library='seaborn')
        
        except Exception as e:
            logging.error(f"Failed to plot scatter matrix: {e}", exc_info=True)
            plt.close('all')
            raise RuntimeError(f"Failed to plot scatter matrix.") from e

    def plot_ridge_plot(self, data: Optional[pd.DataFrame] = None, by: str = 'Intelligence', figsize: Tuple[int, int] = (12, 8), colormap: str = 'viridis', alpha: float = 0.8) -> None:
        logging.debug(f"Starting to plot ridge plot grouped by {by}.")
        
        data = self._load_data(data) if data is not None else self.data
        if data is None:
            raise ValueError("No data provided for plotting.")
        self.clean_data()
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            error_message = "No numeric columns available for ridge plot."
            logging.warning(error_message)
            return
        
        self._validate_columns([by], data=data)
        
        try:
            if isinstance(colormap, str):
                colormap = plt.get_cmap(colormap)
            
            fig, axes = joypy.joyplot(data, by=by, figsize=figsize, colormap=colormap, alpha=alpha, linewidth=1, legend=True)
            plt.title(f'Ridge Plot of Parameters Grouped by {by}', fontsize=16)
            plt.xlabel('Value', fontsize=14)
            plt.ylabel('Parameter', fontsize=14)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.grid(alpha=0.3)
            self._save_and_show_plot('ridge_plots', self._generate_filename(f'ridge_plot_by_{by}'), plot_object=(fig, axes), plot_library='joypy')
        
        except TypeError as e:
            if "'str' object is not callable" in str(e):
                logging.error(f"Colormap error: {e}. Ensure colormap is a valid colormap object.")
                raise ValueError("Invalid colormap provided. Ensure colormap is a valid colormap object.")

    def plot_violinplot(self, data: Optional[pd.DataFrame] = None, x: Optional[str] = None, y: str = 'Intelligence', scale: str = 'width', inner: str = 'quartile', figsize: Tuple[int, int] = (20, 15)) -> None:
        logging.debug(f"Starting to plot violinplot with x={x} and y={y}.")
        data = self._load_data(data) if data is not None else self.data
        if data is None:
            raise ValueError("No data provided for plotting.")
        self.clean_data()
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            error_message = "No numeric columns available for violinplot."
            logging.warning(error_message)
            return 
        
        def plot_violin(x_col: str) -> None:
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

    def plot_3d_scatter(self, data: Optional[pd.DataFrame] = None, x: Optional[str] = None, y: Optional[str] = None, z: Optional[str] = None, color: str = 'blue', marker: str = 'o', figsize: Tuple[int, int] = (10, 8)) -> None:
        logging.debug(f"Starting to plot 3D scatter plot with x={x}, y={y}, and z={z}.")
        
        data = self._load_data(data) if data is not None else self.data
        if data is None:
            raise ValueError("No data provided for plotting.")
        self.clean_data()
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            logging.warning("No numeric columns available for 3D scatter plot.")
            return
        
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
            x = x or numeric_data.columns[0]
            y = y or numeric_data.columns[1]
            z = z or 'Intelligence'
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
        logging.debug("Starting to plot heatmap.")
        
        data = self._load_data(data) if data is not None else self.data
        if data is None:
            raise ValueError("No data provided for plotting.")
        self.clean_data()
        try:
            numeric_data = data.select_dtypes(include=[np.number]).dropna()
            if numeric_data.empty:
                logging.warning("No numeric columns available for correlation matrix.")
                return
            
            correlation_matrix = numeric_data.corr()
            plt.figure(figsize=figsize)
            heatmap = sns.heatmap(correlation_matrix, cmap=cmap, annot=annot, linewidths=linewidths, fmt=fmt, square=square, cbar_kws={'shrink': 0.5})
            heatmap.set_title('Correlation Matrix', fontsize=16)
            heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, fontsize=12)
            heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, fontsize=12)
            self._save_and_show_plot('heatmap', self._generate_filename('correlation_matrix'), plot_object=heatmap, plot_library='seaborn')
        
        except Exception as e:
            logging.error(f"Failed to plot heatmap: {e}", exc_info=True)
            raise RuntimeError("Failed to plot heatmap.") from e

    def plot_pairplot(self, data: Optional[pd.DataFrame] = None, hue: str = 'Intelligence', palette: str = 'viridis', height: float = 2.5, aspect: float = 1, corner: bool = False) -> None:
        logging.debug(f"Starting to plot pairplot with hue '{hue}'.")
        
        data = self._load_data(data) if data is not None else self.data
        if data is None:
            raise ValueError("No data provided for plotting.")
        self.clean_data()
        try:
            numeric_data = data.select_dtypes(include=[np.number]).dropna()
            if numeric_data.empty:
                logging.warning("No numeric columns available for pairplot.")
                return
            
            if hue not in data.columns:
                logging.warning(f"Hue column '{hue}' not found in data. Defaulting to 'Intelligence'.")
                hue = 'Intelligence'
            
            g = sns.pairplot(data, hue=hue, palette=palette, height=height, aspect=aspect, corner=corner, diag_kind='kde', plot_kws={'alpha': 0.7})
            g.fig.suptitle('Pairplot of Parameters and Intelligence', fontsize=16)
            g.fig.subplots_adjust(top=0.9)
            self._save_and_show_plot('pairplot', self._generate_filename('pairplot'), plot_object=g, plot_library='seaborn')
        
        except Exception as e:
            logging.error(f"Failed to plot pairplot: {e}", exc_info=True)
            raise RuntimeError("Failed to plot pairplot.") from e

    def plot_boxplots(self, data: Optional[pd.DataFrame] = None, x: str = 'Intelligence', figsize: Tuple[int, int] = (12, 8)) -> None:
        logging.debug(f"Starting to plot boxplots with x={x}.")
        
        data = self._load_data(data) if data is not None else self.data
        if data is None:
            raise ValueError("No data provided for plotting.")
        self.clean_data()
        numeric_data = data.select_dtypes(include=[np.number]).dropna()
        if numeric_data.empty:
            logging.warning("No numeric columns available for boxplots.")
            return
        
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

    def plot_histograms(self, data: Optional[pd.DataFrame] = None, bins: int = 10, kde: bool = True, color: Optional[str] = None, figsize: Tuple[int, int] = (10, 6)) -> None:
        logging.debug("Starting to plot histograms.")
        
        data = self._load_data(data) if data is not None else self.data
        if data is None:
            raise ValueError("No data provided for plotting.")
        self.clean_data()
        numeric_data = data.select_dtypes(include=[np.number]).dropna()
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

    def plot_jointplot(self, data: Optional[pd.DataFrame] = None, x: Optional[str] = None, y: str = 'Intelligence', kind: str = 'hex', cmap: str = 'Blues', height: float = 8) -> None:
        logging.debug(f"Starting to plot jointplot with x={x} and y={y}.")
        
        data = self._load_data(data) if data is not None else self.data
        if data is None:
            raise ValueError("No data provided for plotting.")
        self.clean_data()
        numeric_data = data.select_dtypes(include=[np.number]).dropna()
        if numeric_data.empty:
            logging.warning("No numeric columns available for jointplot.")
            return
        
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

    def visualize_results(self, data: Optional[pd.DataFrame] = None) -> None:
        logging.debug("Starting to visualize results.")
        
        data = self._load_data(data) if data is not None else self.data
        if data is None:
            raise ValueError("No data provided for visualization.")
        self.clean_data()
        try:
            if not isinstance(data, pd.DataFrame):
                error_message = "The data attribute must be a pandas DataFrame."
                logging.error(error_message)
                raise ValueError(error_message)
            
            numeric_results = data.select_dtypes(include=[np.number]).dropna()
            if numeric_results.empty:
                error_message = "No numeric columns available for visualization."
                logging.warning(error_message)
                return
            
            logging.debug(f"Numeric columns selected for visualization: {numeric_results.columns.tolist()}")

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

# Example usage for testing
if __name__ == "__main__":
    # Load some example data
    data = pd.read_csv('/home/lloyd/results_chunk_1.csv')

    visualizer = DataVisualizer(data=data)

    # Test each method independently
    try:
        visualizer.plot_scatter_matrix()
        visualizer.plot_ridge_plot()
        visualizer.plot_violinplot()
        visualizer.plot_3d_scatter()
        visualizer.plot_heatmap()
        visualizer.plot_pairplot()
        visualizer.plot_jointplot()
        visualizer.plot_histograms()
        visualizer.plot_boxplots()
    except Exception as e:
        logging.error(f"An error occurred during plotting: {e}")
