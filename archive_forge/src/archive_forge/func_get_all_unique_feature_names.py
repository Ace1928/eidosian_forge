import torch
from typing import Any, Set, Dict, List, Tuple, OrderedDict
from collections import OrderedDict as OrdDict
def get_all_unique_feature_names(self, plottable_features_only: bool=True) -> Set[str]:
    """
        The purpose of this method is to provide a user the set of all feature names so that if
        they wish to use the filtering capabilities of the generate_table_view(), or use either of
        the generate_plot_view() or generate_histogram_view(), they don't need to manually parse
        the generated_reports dictionary to get this information.

        Args:
            plottable_features_only (bool): True if the user is only looking for plottable features,
                False otherwise
                plottable features are those that are tensor values
                Default: True (only return those feature names that are plottable)

        Returns all the unique module fqns present in the reports the ModelReportVisualizer
        instance was initialized with.
        """
    unique_feature_names = set()
    for module_fqn in self.generated_reports:
        feature_dict: Dict[str, Any] = self.generated_reports[module_fqn]
        for feature_name in feature_dict:
            if not plottable_features_only or type(feature_dict[feature_name]) == torch.Tensor:
                unique_feature_names.add(feature_name)
    return unique_feature_names