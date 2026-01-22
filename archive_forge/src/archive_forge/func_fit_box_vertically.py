from typing import Any, Optional, Tuple, Union
@classmethod
def fit_box_vertically(cls, left: Optional[float]=None) -> 'Fit':
    """
        Display the page designated by page, with the horizontal coordinate
        left positioned at the left edge of the window and the contents of the
        page magnified just enough to fit the entire height of its bounding box
        within the window.

        A null value for left specifies that the current value of that
        parameter is to be retained unchanged.

        Args:
            left:

        Returns:
            The created fit object.
        """
    return Fit(fit_type='/FitBV', fit_args=(left,))