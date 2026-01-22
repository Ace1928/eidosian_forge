import csv
from config import global_config
from logger import CustomLogger
def export_weights(self, weights, filename='network_weights.csv'):
    """
        Exports the weights of the neural network to a CSV file.

        Parameters:
            weights (dict): The weights of the neural network to export.
            filename (str): The name of the file to export the weights to.
        """
    try:
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file, delimiter=self.config['delimiter'], quotechar=self.config['quotechar'], quoting=csv.QUOTE_MINIMAL)
            for layer, weight_matrix in weights.items():
                writer.writerow([layer])
                for row in weight_matrix:
                    writer.writerow(row)
        self.logger.info(f'Weights successfully exported to {filename}.')
    except Exception as e:
        self.logger.error(f'Failed to export weights to {filename}. Error: {e}')