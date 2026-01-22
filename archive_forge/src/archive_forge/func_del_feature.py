from ._Feature import Feature
import re
def del_feature(self, feature_id):
    """Delete a feature.

        Arguments:
         - feature_id: Unique id of the feature to delete

        Remove a feature from the set, indicated by its id.
        """
    del self.features[feature_id]