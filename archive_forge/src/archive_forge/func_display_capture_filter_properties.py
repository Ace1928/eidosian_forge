import pygame
def display_capture_filter_properties(self):
    """Displays a dialog containing the property page of the capture filter.

        For VfW drivers you may find the option to select the resolution most
        likely here.
        """
    self.dev.displaycapturefilterproperties()