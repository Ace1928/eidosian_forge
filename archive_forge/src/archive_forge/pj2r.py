import sys
from config import global_config
from logger import CustomLogger
from fractal_generator import FractalGenerator
from neural_network import NeuralNetwork
from visualization import visualize_network
from csv_exporter import CSVExporter


def main():
    # Initialize logger
    logger = CustomLogger("Main").get_logger()

    try:
        # Generate the fractal neural network structure
        logger.info("Generating fractal neural network structure...")
        fractal_generator = FractalGenerator(global_config.base_layer_hexagons)
        network_structure = fractal_generator.generate_network()

        # Initialize the neural network with the generated structure
        logger.info("Initializing neural network...")
        neural_network = NeuralNetwork(network_structure)

        # Visualize the neural network
        logger.info("Visualizing neural network...")
        visualize_network(network_structure, global_config.visualization_options)

        # Export the neural network structure to CSV
        logger.info("Exporting neural network structure to CSV...")
        csv_exporter = CSVExporter()
        csv_exporter.export_network_structure(network_structure)

        logger.info("Process completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
