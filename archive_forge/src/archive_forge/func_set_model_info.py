from Bio.PDB.StructureBuilder import StructureBuilder
import numpy as np
def set_model_info(self, model_id, chain_count):
    """Set the information for a model.

        :param model_id: the index for the model
        :param chain_count: the number of chains in the model

        """
    self.structure_builder.init_model(model_id)