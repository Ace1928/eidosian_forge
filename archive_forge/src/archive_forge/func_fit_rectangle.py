from typing import Any, Optional, Tuple, Union
@classmethod
def fit_rectangle(cls, left: Optional[float]=None, bottom: Optional[float]=None, right: Optional[float]=None, top: Optional[float]=None) -> 'Fit':
    """
        Display the page designated by page , with its contents magnified
        just enough to fit the rectangle specified by the coordinates
        left, bottom, right, and top entirely within the window
        both horizontally and vertically.

        If the required horizontal and vertical magnification factors are
        different, use the smaller of the two, centering the rectangle within
        the window in the other dimension.

        A null value for any of the parameters may result in unpredictable
        behavior.

        Args:
            left:
            bottom:
            right:
            top:

        Returns:
            The created fit object.
        """
    return Fit(fit_type='/FitR', fit_args=(left, bottom, right, top))