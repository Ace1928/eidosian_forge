#!/usr/bin/env python3
"""
Professional Voice Editor with Advanced DSP
===========================================

A high-quality audio processing application featuring:
- 24-bit/96kHz recording capability
- Multi-band parametric EQ
- AI-based noise reduction
- Broadcast-quality compression
- Real-time waveform & spectrum analysis
- Non-destructive effect stacking
- File I/O support
- Region selection editing
- Professional audio effects chain
"""

import sys  # Import the sys module for system-specific parameters and functions.
import logging  # Import the logging module for logging events during program execution.
import wave  # Import the wave module for reading and writing WAV files.
import io  # Import the io module for working with various types of I/O streams.
import numpy as np  # Import the numpy module for numerical operations, especially with arrays.
import sounddevice as sd  # Import the sounddevice module for audio input and output.
from scipy.signal import (
    sosfiltfilt,  # Import the sosfiltfilt function for applying second-order section filters.
    butter,  # Import the butter function for designing Butterworth filters.
    cheby2,  # Import the cheby2 function for designing Chebyshev type II filters.
    lfilter,  # Import the lfilter function for applying digital filters.
    sosfilt,  # Import the sosfilt function for applying second-order section filters.
)  # Import specific signal processing functions from scipy.signal.
from PyQt5 import (
    QtCore,  # Import QtCore for core non-GUI functionality in PyQt.
    QtWidgets,  # Import QtWidgets for GUI elements in PyQt.
    QtGui,  # Import QtGui for graphical elements in PyQt.
)  # Import necessary modules from PyQt5 for GUI development.
from PyQt5.QtCore import pyqtSignal  # Import pyqtSignal for custom signals in PyQt.
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,  # Import FigureCanvas for embedding matplotlib plots in PyQt.
)  # Import FigureCanvas for embedding matplotlib plots in PyQt.
from matplotlib.figure import Figure  # Import Figure for creating matplotlib figures.
import pyqtgraph as pg  # Import pyqtgraph for high-performance plotting and data visualization.
import noisereduce as nr  # Import noisereduce for noise reduction in audio signals.
import pyrubberband  # Import pyrubberband for time stretching and pitch shifting of audio.
import librosa  # Import librosa for audio analysis and manipulation.
from pydub import (
    AudioSegment,  # Import AudioSegment from pydub for audio file manipulation.
)  # Import AudioSegment from pydub for audio file manipulation.
from pydub.playback import play  # Import play from pydub.playback for audio playback.
from librosa import util  # Import util from librosa for utility functions.
import threading  # Import the threading module for concurrent execution of code.
import os  # Import the os module for interacting with the operating system.
from collections import (
    deque,  # Import deque from collections for double-ended queue implementation.
)  # Import deque from collections for double-ended queue implementation.
from typing import (
    Dict,  # Import Dict for type hinting dictionaries.
    List,  # Import List for type hinting lists.
    Optional,  # Import Optional for type hinting optional values.
    Any,  # Import Any for type hinting values of any type.
    Deque,  # Import Deque for type hinting double-ended queues.
    Tuple,  # Import Tuple for type hinting tuples.
)  # Import typing hints for static type checking.

# Configure professional audio settings
AUDIO_CONFIG: Dict[str, Any] = {
    "sample_rate": 96000,  # Sample rate in Hz, defining the number of audio samples per second.
    "bit_depth": 24,  # Bit depth of audio, determining the dynamic range of the audio signal.
    "channels": 1,  # Number of audio channels, set to 1 for mono audio.
    "dtype": "float32",  # Data type for audio samples, using 32-bit floating-point for precision.
    "blocksize": 2048,  # Block size for audio processing, the number of samples processed at once.
    "latency": "high",  # Audio latency setting, set to 'high' for minimal delay.
}
logger = logging.getLogger(__name__)
# Configure logging for the application
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO, capturing informational messages and above.
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",  # Define the log message format, including timestamp, level, logger name, and message.
    datefmt="%Y-%m-%d %H:%M:%S",  # Define the date format
)
logger.info(
    "Application logging initialized."
)  # Log that the application logging has been initialized.


class EffectRegistry:
    """
    A registry to store and retrieve audio effects processors.
    """

    def __init__(self) -> None:
        """
        Initializes the EffectRegistry with an empty dictionary to store effects.
        """
        self._effects: Dict[str, Any] = {}  # Dictionary to hold effect processors

    def register(self, name: str, processor: Any) -> None:
        """
        Registers an effect processor with a given name.

        Args:
            name (str): The name of the effect.
            processor (Any): The effect processor function.
        """
        logger.debug(
            f"Registering effect: {name}"
        )  # Log the registration of an effect.
        self._effects[name] = processor  # Store the processor in the dictionary

    def get(self, name: str) -> Optional[Any]:
        """
        Retrieves an effect processor by its name.

        Args:
            name (str): The name of the effect.

        Returns:
            Optional[Any]: The effect processor function, or None if not found.
        """
        logger.debug(f"Getting effect: {name}")  # Log the retrieval of an effect.
        return self._effects.get(name)  # Return the processor if it exists


class ProfessionalAudioEngine:
    """High-performance audio processing engine with DSP effects"""

    def __init__(self) -> None:
        """
        Initializes the audio engine with an empty effect chain, sample rate, and effect registry.
        """
        self.effect_chain: List[Dict[str, Any]] = []  # List to hold the audio effects
        self.sample_rate: int = AUDIO_CONFIG["sample_rate"]  # Sample rate from config
        self.noise_profile: Optional[np.ndarray] = (
            None  # Noise profile for noise reduction
        )
        self.effect_registry: EffectRegistry = EffectRegistry()  # Registry for effects
        self._register_default_effects()  # Register default effects

    def _register_default_effects(self) -> None:
        """
        Registers the default audio effects with the effect registry.
        """
        logger.debug(
            "Registering default effects"
        )  # Log the registration of default effects.
        self.effect_registry.register("eq", self._apply_eq)  # Register EQ effect
        self.effect_registry.register(
            "compressor", self._apply_compression
        )  # Register compressor effect
        self.effect_registry.register(
            "denoise", self._apply_denoise
        )  # Register denoise effect
        self.effect_registry.register(
            "pitch", self._apply_pitch
        )  # Register pitch effect
        self.effect_registry.register(
            "bandpass", self._apply_bandpass
        )  # Register bandpass effect

    def add_effect(self, effect_type: str, parameters: Dict[str, Any]) -> None:
        """
        Adds an effect to the effect chain.

        Args:
            effect_type (str): The type of the effect.
            parameters (Dict[str, Any]): The parameters for the effect.
        """
        logger.debug(
            f"Adding effect: {effect_type} with params: {parameters}"
        )  # Log the addition of an effect.
        effect: Dict[str, Any] = {
            "type": effect_type,  # Store the effect type.
            "params": parameters,  # Store the effect parameters.
            "enabled": True,  # Set the effect to enabled by default.
        }  # Create effect dictionary
        self.effect_chain.append(effect)  # Add effect to the chain

    def process_audio(self, data: Optional[np.ndarray]) -> np.ndarray:
        """
        Processes the audio data through the effect chain.

        Args:
            data (Optional[np.ndarray]): The input audio data.

        Returns:
            np.ndarray: The processed audio data.
        """
        logger.debug(
            f"Processing audio data of length: {len(data) if data is not None else None}"
        )  # Log the start of audio processing.
        # Add comprehensive input validation
        if data is None or len(data) < 64:  # Minimum processing block size
            logger.warning(
                f"Invalid audio block size: {len(data) if data else None}"
            )  # Log a warning for invalid audio block size.
            return (
                np.array([]) if data is None else data.copy()
            )  # Return empty or original data

        processed: np.ndarray = data.copy()  # Copy the input data
        for effect in self.effect_chain:  # Iterate through the effect chain
            if not effect["enabled"]:  # Skip if effect is disabled
                logger.debug(
                    f"Skipping disabled effect: {effect['type']}"
                )  # Log skipping of disabled effect.
                continue
            processor = self.effect_registry.get(
                effect["type"]
            )  # Get the effect processor
            if processor:
                try:
                    # Add safe processing guard
                    logger.debug(
                        f"Applying effect: {effect['type']}"
                    )  # Log the application of an effect.
                    processed = processor(
                        processed, **effect["params"]
                    )  # Apply the effect
                except Exception as e:
                    logger.error(
                        f"Effect {effect['type']} failed: {str(e)}"
                    )  # Log error if effect fails
                    continue  # Continue to next effect
        logger.debug(
            "Audio processing complete"
        )  # Log the completion of audio processing.
        return processed  # Return the processed audio

    def _apply_eq(
        self,
        data: np.ndarray,
        low_shelf: float = 0,
        mid_boost: float = 0,
        high_shelf: float = 0,
    ) -> np.ndarray:
        """
        Applies a multi-band parametric EQ to the audio data.

        Args:
            data (np.ndarray): The input audio data.
            low_shelf (float): The gain for the low shelf filter in dB.
            mid_boost (float): The gain for the mid boost filter in dB.
            high_shelf (float): The gain for the high shelf filter in dB.

        Returns:
            np.ndarray: The EQ processed audio data.
        """
        logger.debug(
            f"Applying EQ with low_shelf: {low_shelf}, mid_boost: {mid_boost}, high_shelf: {high_shelf}"
        )  # Log the application of the EQ effect.
        # Add buffer length validation
        if len(data) < 10:  # Minimum length for SOS filtering
            logger.warning(
                "Input data too short for EQ processing"
            )  # Log a warning for short input data.
            return data  # Return original data if too short

        sos_low: np.ndarray = butter(
            2, 200, "lowpass", fs=self.sample_rate, output="sos"
        )  # Low shelf filter
        sos_high: np.ndarray = butter(
            2, 5000, "highpass", fs=self.sample_rate, output="sos"
        )  # High shelf filter

        # Process only if buffer is sufficiently long
        processed: np.ndarray = np.zeros_like(data)  # Initialize processed data
        if len(data) > sos_low.shape[0] * 6:  # Required by sosfiltfilt
            processed += sosfiltfilt(sos_low, data) * (
                10 ** (low_shelf / 20)
            )  # Apply low shelf
        if len(data) > sos_high.shape[0] * 6:
            processed += sosfiltfilt(sos_high, data) * (
                10 ** (high_shelf / 20)
            )  # Apply high shelf

        mid: np.ndarray = cheby2(
            4, 40, [1000, 4000], "bandpass", fs=self.sample_rate, output="sos"
        )  # Mid boost filter
        if len(data) > mid.shape[0] * 6:
            processed += sosfiltfilt(mid, data) * (
                10 ** (mid_boost / 20)
            )  # Apply mid boost

        logger.debug("EQ processing complete")  # Log the completion of EQ processing.
        return processed * 0.7  # Return processed data with gain adjustment

    def _apply_compression(
        self,
        data: np.ndarray,
        threshold: float = -20,
        ratio: float = 4,
        attack: float = 0.01,
        release: float = 0.1,
    ) -> np.ndarray:
        """
        Applies dynamic range compression to the audio data.

        Args:
            data (np.ndarray): The input audio data.
            threshold (float): The threshold in dB.
            ratio (float): The compression ratio.
            attack (float): The attack time in seconds.
            release (float): The release time in seconds.

        Returns:
            np.ndarray: The compressed audio data.
        """
        logger.debug(
            f"Applying compression with threshold: {threshold}, ratio: {ratio}, attack: {attack}, release: {release}"
        )  # Log the application of the compression effect.
        envelope: np.ndarray = util.normalize(
            np.abs(data), norm=np.inf
        )  # Calculate the envelope
        gain_reduction: np.ndarray = np.zeros_like(
            envelope
        )  # Initialize gain reduction array

        attack_coeff: float = np.exp(
            -1 / (attack * self.sample_rate)
        )  # Attack coefficient
        release_coeff: float = np.exp(
            -1 / (release * self.sample_rate)
        )  # Release coefficient

        for i in range(1, len(envelope)):  # Iterate through the envelope
            if envelope[i] > threshold:  # If envelope exceeds threshold
                target: float = (
                    threshold + (envelope[i] - threshold) / ratio
                )  # Calculate target gain
                coeff: float = (
                    attack_coeff
                    if target < gain_reduction[i - 1]
                    else release_coeff  # Choose attack or release coefficient
                )
            else:
                target = 1.0  # Target gain is 1 if below threshold
                coeff = release_coeff  # Use release coefficient

            gain_reduction[i] = (1 - coeff) * target + coeff * gain_reduction[
                i - 1
            ]  # Calculate gain reduction
        logger.debug(
            "Compression processing complete"
        )  # Log the completion of compression processing.
        return data * gain_reduction  # Apply gain reduction to the audio

    def _apply_denoise(
        self, data: np.ndarray, reduction_db: float = 20, stationary: bool = True
    ) -> np.ndarray:
        """
        Applies noise reduction to the audio data.

        Args:
            data (np.ndarray): The input audio data.
            reduction_db (float): The amount of noise reduction in dB.
            stationary (bool): Whether the noise is stationary.

        Returns:
            np.ndarray: The denoised audio data.
        """
        logger.debug(
            f"Applying denoise with reduction_db: {reduction_db}, stationary: {stationary}"
        )  # Log the application of the denoise effect.
        if self.noise_profile is None:  # If no noise profile is available
            logger.warning(
                "No noise profile available for denoise"
            )  # Log a warning if no noise profile is available.
            return data  # Return original data

        denoised_data: np.ndarray = nr.reduce_noise(  # Apply noise reduction
            y=data,
            sr=self.sample_rate,
            y_noise=self.noise_profile,
            prop_decrease=reduction_db / 40,
            stationary=stationary,
        )
        logger.debug(
            "Denoise processing complete"
        )  # Log the completion of denoise processing.
        return denoised_data  # Return denoised data

    def _apply_pitch(self, data: np.ndarray, factor: float) -> np.ndarray:
        """
        Applies pitch shifting to the audio data.

        Args:
            data (np.ndarray): The input audio data.
            factor (float): The pitch shift factor.

        Returns:
            np.ndarray: The pitch-shifted audio data.
        """
        logger.debug(
            f"Applying pitch shift with factor: {factor}"
        )  # Log the application of the pitch shift effect.
        try:
            pitched_data: np.ndarray = pyrubberband.time_stretch(
                data, self.sample_rate, factor
            )  # Apply pitch shift
            logger.debug(
                "Pitch shifting complete"
            )  # Log the completion of pitch shifting.
            return pitched_data  # Return pitch-shifted data
        except Exception as e:
            logger.error(
                f"Pitch shifting failed: {str(e)}"
            )  # Log error if pitch shift fails
            logger.warning(
                "Ensure rubberband-cli is installed: 'brew install rubberband' or 'apt-get install rubberband-cli'"
            )  # Log a warning about rubberband installation.
            return data  # Return original audio on failure

    def _apply_bandpass(
        self, data: np.ndarray, lowcut: float = 300, highcut: float = 3400
    ) -> np.ndarray:
        """
        Applies a bandpass filter to the audio data.

        Args:
            data (np.ndarray): The input audio data.
            lowcut (float): The low cutoff frequency in Hz.
            highcut (float): The high cutoff frequency in Hz.

        Returns:
            np.ndarray: The bandpass filtered audio data.
        """
        logger.debug(
            f"Applying bandpass filter with lowcut: {lowcut}, highcut: {highcut}"
        )  # Log the application of the bandpass filter.
        nyq: float = 0.5 * self.sample_rate  # Nyquist frequency
        low: float = lowcut / nyq  # Normalized low cutoff
        high: float = highcut / nyq  # Normalized high cutoff
        sos: np.ndarray = butter(
            5, [low, high], btype="band", output="sos"
        )  # Create bandpass filter
        filtered_data: np.ndarray = sosfilt(sos, data)  # Apply bandpass filter
        logger.debug(
            "Bandpass filter complete"
        )  # Log the completion of bandpass filtering.
        return filtered_data  # Return filtered data


class SpectrumAnalyzer(FigureCanvas):
    """
    A Matplotlib canvas for displaying the audio spectrum, providing real-time frequency analysis.
    This class visualizes the frequency content of audio signals, including a peak hold feature.
    It inherits from `FigureCanvas` to integrate with PyQt5 applications.
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        """
        Initializes the SpectrumAnalyzer with a Matplotlib figure and axes, setting up the visual
        representation for frequency analysis.

        Args:
            parent (Optional[QtWidgets.QWidget]): The parent widget for this canvas, allowing
                integration into a larger PyQt5 application. Defaults to None if not specified.
        """
        logger.debug(
            "Initializing SpectrumAnalyzer"
        )  # Log the start of SpectrumAnalyzer initialization.
        self.fig: Figure = Figure(
            figsize=(8, 3), dpi=100
        )  # Create a Matplotlib figure with specified size and DPI.
        super().__init__(
            self.fig
        )  # Initialize the FigureCanvas with the created Matplotlib figure.
        self.ax = self.fig.add_subplot(
            111
        )  # Add a single subplot to the figure for the spectrum display.
        self.ax.set_facecolor(
            "#1e1e1e"
        )  # Set the background color of the axes to a dark gray.
        self.fig.set_facecolor(
            "#2d2d2d"
        )  # Set the background color of the entire figure to a slightly lighter gray.
        self.ax.tick_params(
            colors="#888888"
        )  # Set the color of the tick marks on the axes to a light gray.
        self.ax.xaxis.label.set_color(
            "#888888"
        )  # Set the color of the x-axis label to a light gray.
        self.ax.yaxis.label.set_color(
            "#888888"
        )  # Set the color of the y-axis label to a light gray.
        self.fig.tight_layout()  # Adjust the layout of the figure to ensure all elements fit without overlapping.

        (self.spectrum_line,) = self.ax.semilogx(
            [], [], color="cyan", linewidth=1.2
        )  # Initialize an empty line plot for the spectrum, using a semilogarithmic x-axis, cyan color, and a slightly thicker line.
        self.ax.set_xlim(
            20, 20000
        )  # Set the x-axis limits to display frequencies from 20 Hz to 20 kHz.
        self.ax.set_ylim(
            -90, 0
        )  # Set the y-axis limits to display decibel values from -90 dBFS to 0 dBFS.
        self.ax.set_xlabel(
            "Frequency (Hz)"
        )  # Set the label for the x-axis to "Frequency (Hz)".
        self.ax.set_ylabel(
            "dBFS"
        )  # Set the label for the y-axis to "dBFS" (decibels relative to full scale).

        # Add peak hold line
        (self.peak_line,) = self.ax.semilogx(
            [], [], color="magenta", alpha=0.7, linewidth=0.8, linestyle="--"
        )  # Initialize an empty dashed line plot for the peak hold, using a semilogarithmic x-axis, magenta color, transparency, and a thinner line.
        self.peak_values: Optional[np.ndarray] = (
            None  # Initialize the peak values array to None, to be updated during spectrum updates.
        )
        self.fft_length: int = (
            4096  # Set a fixed FFT length for consistent spectrum analysis.
        )

    def update_spectrum(self, data: np.ndarray, sr: int) -> None:
        """
        Updates the spectrum display with new audio data, performing FFT analysis and updating
        both the real-time spectrum and the peak hold display.

        Args:
            data (np.ndarray): The input audio data as a NumPy array. It is expected to be a 1D array.
            sr (int): The sample rate of the audio data, used for frequency calculations.
        """
        logger.debug(
            f"Updating spectrum with data of length: {len(data)}, sample rate: {sr}"
        )  # Log the start of spectrum update with data length and sample rate.
        # Ensure 1D input and limit to valid audio range
        data = np.asarray(
            data
        ).squeeze()  # Ensure the input data is a 1D NumPy array by squeezing it.
        data = data[: self.fft_length]  # Limit the data to the fixed FFT length.
        data = np.clip(
            data, -1, 1
        )  # Clip the audio data to the range [-1, 1] to prevent invalid values.

        fft_length: int = len(
            data
        )  # Get the actual length of the data, which may be less than self.fft_length if the input is shorter.
        if fft_length < 2:  # Check if the data length is too short for FFT analysis.
            logger.warning(
                "Input data too short for spectrum analysis"
            )  # Log a warning if the data is too short.
            return  # Exit the function if the data is too short.

        valid_data: np.ndarray = data  # Assign the data to a new variable for clarity.
        window: np.ndarray = np.hanning(
            fft_length
        )  # Create a Hanning window of the same length as the data.

        # Apply FFT with proper normalization
        fft: np.ndarray = np.fft.rfft(
            valid_data * window
        )  # Apply the real-valued FFT to the windowed data.
        freq: np.ndarray = np.fft.rfftfreq(
            fft_length, d=1 / sr
        )  # Calculate the corresponding frequencies for the FFT output.
        db: np.ndarray = 20 * np.log10(
            np.abs(fft) + 1e-9
        )  # Calculate the magnitude of the FFT in decibels.
        db = np.clip(
            db - np.max(db), -90, 0
        )  # Normalize the dB values and clip them to the range [-90, 0].

        self.spectrum_line.set_data(
            freq, db
        )  # Update the spectrum line data with the calculated frequencies and dB values.
        self.ax.relim()  # Reset the axes limits to fit the new data.
        self.ax.autoscale_view(
            scalex=False, scaley=True
        )  # Autoscale the view, keeping the x-axis scale fixed.
        self.draw()  # Redraw the canvas to display the updated spectrum.

        # Update peak hold
        if self.peak_values is None or len(self.peak_values) != len(
            db
        ):  # Check if the peak values array is not initialized or if its length does not match the current dB values.
            self.peak_values = (
                db.copy()
            )  # Initialize the peak values array with the current dB values.
        else:
            self.peak_values = np.maximum(
                self.peak_values, db
            )  # Update the peak values by taking the maximum of the current peak values and the new dB values.

        self.peak_line.set_data(
            freq, self.peak_values
        )  # Update the peak hold line data with the calculated frequencies and peak values.
        logger.debug("Spectrum updated")  # Log the completion of the spectrum update.


import copy
from typing import List, Dict, Any, Optional
from collections import deque
import numpy as np
from PyQt5 import QtWidgets, QtCore
import sounddevice as sd
from pydub import AudioSegment
import pyqtgraph as pg
import threading
import logging
import io
import wave
import os
from PyQt5.QtCore import pyqtSignal

# Configure logging for the module
logger = logging.getLogger(__name__)

# Placeholder for audio configuration, replace with actual config
AUDIO_CONFIG = {
    "sample_rate": 48000,
    "channels": 1,
    "dtype": "float32",
    "blocksize": 1024,
}


class HistoryManager:
    """
    Manages the history of application states for undo/redo functionality.
    Implements a stack-based approach with a maximum history limit.
    """

    MAX_HISTORY: int = 50  # Maximum number of states to store in history

    def __init__(self) -> None:
        """
        Initializes the HistoryManager with an empty stack and a position pointer.
        The stack stores application states as dictionaries, and the position
        indicates the current state within the history.
        """
        self.stack: List[Dict[str, Any]] = []  # Stack to hold application states
        self.position: int = -1  # Current position in the stack, -1 indicates no state

    def push_state(self, state: Dict[str, Any]) -> None:
        """
        Pushes a new application state onto the history stack.
        If the current position is not at the end of the stack, the stack is truncated
        to remove any future states before appending the new state. This ensures
        that branching history is not created.

        Args:
            state (Dict[str, Any]): The application state to push onto the stack.
        """
        logger.debug("Pushing new state to history")
        if self.position < len(self.stack) - 1:  # Check if not at the end of the stack
            self.stack = self.stack[
                : self.position + 1
            ]  # Truncate the stack to current position
        self.stack.append(copy.deepcopy(state))  # Append a deep copy of the new state
        self.position = min(
            self.position + 1, self.MAX_HISTORY - 1
        )  # Increment position, capped by MAX_HISTORY
        logger.debug(
            f"Current history position: {self.position}, stack size: {len(self.stack)}"
        )


class ProfessionalVoiceEditor(QtWidgets.QMainWindow):
    """
    Main application window for the Professional Voice Editor.
    This class manages the main window, audio processing, UI elements, and application state.
    It inherits from QtWidgets.QMainWindow to provide standard window functionalities.
    """

    # Define signals at class level
    update_signal: pyqtSignal = (
        QtCore.pyqtSignal()
    )  # Signal for updating displays, emitted when audio processing occurs.

    def __init__(self) -> None:
        """
        Initializes the main application window.
        This constructor sets up the window, initializes audio engine, UI components,
        audio stream, timer, effect toggles, styling, history manager, and audio lock.
        """
        logger.info("Initializing ProfessionalVoiceEditor: Starting main window setup.")
        super().__init__()  # Initialize QMainWindow, calling the parent class constructor.
        logger.debug("QMainWindow initialized.")

        self.setWindowTitle(
            "Professional Voice Editor"
        )  # Set window title, displayed in the window's title bar.
        logger.debug("Window title set to 'Professional Voice Editor'.")

        self.setGeometry(
            100, 100, 1600, 900
        )  # Set window geometry, defining the window's position and size.
        logger.debug("Window geometry set to (100, 100, 1600, 900).")

        # Audio state
        logger.debug("Initializing audio state variables.")
        self.engine: ProfessionalAudioEngine = (
            ProfessionalAudioEngine()
        )  # Initialize audio engine, responsible for audio processing.
        logger.debug("ProfessionalAudioEngine initialized.")
        self.raw_audio_segment: Optional[AudioSegment] = (
            None  # Raw audio segment, stores the loaded audio data.
        )
        logger.debug("raw_audio_segment initialized to None.")
        self.proc_audio_segment: Optional[AudioSegment] = (
            None  # Processed audio segment, stores the audio data after applying effects.
        )
        logger.debug("proc_audio_segment initialized to None.")
        self.audio_data: Optional[np.ndarray] = None  # Raw audio data as a NumPy array.
        logger.debug("audio_data initialized to None.")
        self.processed_data: Optional[np.ndarray] = (
            None  # Processed audio data as a NumPy array.
        )
        logger.debug("processed_data initialized to None.")
        self.is_recording: bool = (
            False  # Recording state, indicates if the application is currently recording audio.
        )
        logger.debug("is_recording initialized to False.")
        self.recording_buffer: Deque[np.ndarray] = deque(
            maxlen=100
        )  # Fixed-size buffer for recording, stores incoming audio data during recording.
        logger.debug("recording_buffer initialized with maxlen=100.")
        self.sample_rate: int = AUDIO_CONFIG[
            "sample_rate"
        ]  # Sample rate from config, the number of audio samples per second.
        logger.debug(
            f"sample_rate initialized to {self.sample_rate} from AUDIO_CONFIG."
        )
        self.start_idx: int = (
            0  # Start index for region selection, used for selecting a portion of the audio.
        )
        logger.debug("start_idx initialized to 0.")
        self.end_idx: int = (
            0  # End index for region selection, used for selecting a portion of the audio.
        )
        logger.debug("end_idx initialized to 0.")

        # UI components
        logger.debug("Initializing UI components.")
        self.waveform: pg.PlotWidget = (
            pg.PlotWidget()
        )  # Waveform plot widget, displays the audio waveform.
        logger.debug("waveform plot widget initialized.")
        self.spectrum: SpectrumAnalyzer = (
            SpectrumAnalyzer()
        )  # Spectrum analyzer widget, displays the frequency spectrum of the audio.
        logger.debug("spectrum analyzer widget initialized.")
        self.meter: pg.PlotWidget = (
            pg.PlotWidget()
        )  # Meter plot widget, displays the RMS level of the audio.
        logger.debug("meter plot widget initialized.")
        self._init_ui()  # Initialize UI, calls the method to set up the user interface.
        logger.debug("UI initialization complete.")

        # Audio stream
        logger.debug("Initializing audio stream.")
        self.stream: sd.InputStream = sd.InputStream(
            samplerate=AUDIO_CONFIG["sample_rate"],
            channels=AUDIO_CONFIG["channels"],
            dtype=AUDIO_CONFIG["dtype"],
            blocksize=AUDIO_CONFIG["blocksize"],
            callback=self._audio_callback,
        )  # Initialize audio stream, sets up the input stream for audio recording.
        logger.debug("Audio stream initialized.")

        # Timer for updates
        logger.debug("Initializing timer for updates.")
        self.timer: QtCore.QTimer = (
            QtCore.QTimer()
        )  # Timer for periodic updates, used to refresh the UI.
        logger.debug("QTimer initialized.")
        self.timer.timeout.connect(
            self._update_displays
        )  # Connect timer to update displays, connects the timer's timeout signal to the _update_displays method.
        logger.debug("Timer timeout connected to _update_displays.")
        self.timer.start(50)  # Start timer, starts the timer with a 50ms interval.
        logger.debug("Timer started with 50ms interval.")

        # Add effect enable toggles
        logger.debug("Initializing effect enable toggles.")
        self.active_effects: Dict[str, bool] = {
            "eq": True,
            "compressor": True,
            "denoise": True,
            "pitch": False,
            "bandpass": False,
        }  # Dictionary to hold active effects, stores the state of each effect.
        logger.debug(f"Active effects initialized: {self.active_effects}")

        # Apply modern dark theme and styling
        logger.debug("Applying dark theme and styling.")
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #2d2d2d;
                color: #cccccc;
            }
            QToolBar {
                background-color: #353535;
                border: none;
                padding: 4px;
            }
            QPushButton {
                background-color: #404040;
                border: 1px solid #505050;
                border-radius: 4px;
                padding: 5px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #505050;
            }
            QSlider::groove:horizontal {
                height: 8px;
                background: #404040;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                width: 16px;
                margin: -4px 0;
                background: #808080;
                border-radius: 8px;
            }
            QTabWidget::pane {
                border: 1px solid #404040;
                background: #353535;
            }
            QTabBar::tab {
                background: #404040;
                color: #cccccc;
                padding: 8px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background: #505050;
            }
        """
        )  # Set application style, applies a dark theme to the application.
        logger.debug("Dark theme and styling applied.")

        self.history: HistoryManager = (
            HistoryManager()
        )  # Initialize history manager, used for undo/redo functionality.
        logger.debug("HistoryManager initialized.")
        self._init_undo_redo()  # Initialize undo/redo actions, sets up the undo/redo functionality.
        logger.debug("Undo/redo actions initialized.")

        self.update_signal.connect(
            self._update_displays
        )  # Connect update signal to update displays, connects the custom update signal to the _update_displays method.
        logger.debug("update_signal connected to _update_displays.")

        self.audio_lock: threading.Lock = (
            threading.Lock()
        )  # Lock for audio processing, used to synchronize access to audio data.
        logger.debug("Audio lock initialized.")
        logger.info("ProfessionalVoiceEditor initialization complete.")

    def _init_ui(self) -> None:
        """
        Initializes the user interface components.
        This method creates the main widget, sets it as the central widget,
        creates the main layout, toolbar, visualization container, and effect panel.
        """
        logger.debug("Initializing UI: Starting UI component setup.")
        main_widget: QtWidgets.QWidget = (
            QtWidgets.QWidget()
        )  # Create main widget, the base widget for the application.
        logger.debug("Main widget created.")
        self.setCentralWidget(
            main_widget
        )  # Set central widget, sets the main widget as the central widget of the main window.
        logger.debug("Main widget set as central widget.")
        layout: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout(
            main_widget
        )  # Create main layout, a vertical box layout for the main widget.
        logger.debug("Main layout created.")

        # Toolbar
        logger.debug("Creating toolbar.")
        toolbar: QtWidgets.QToolBar = (
            QtWidgets.QToolBar()
        )  # Create toolbar, a toolbar for common actions.
        logger.debug("Toolbar created.")
        self._create_toolbar(
            toolbar
        )  # Create toolbar buttons, calls the method to add buttons to the toolbar.
        logger.debug("Toolbar buttons created.")
        layout.addWidget(
            toolbar
        )  # Add toolbar to layout, adds the toolbar to the main layout.
        logger.debug("Toolbar added to layout.")

        # Visualization
        logger.debug("Creating visualization container.")
        vis_container: QtWidgets.QSplitter = QtWidgets.QSplitter(
            QtCore.Qt.Orientation.Vertical
        )  # Create splitter for visualizations, a splitter to arrange visualization widgets.
        logger.debug("Visualization container created.")
        self._create_visualizations(
            vis_container
        )  # Create visualization widgets, calls the method to add visualization widgets to the container.
        logger.debug("Visualization widgets created.")
        layout.addWidget(
            vis_container
        )  # Add visualization container to layout, adds the visualization container to the main layout.
        logger.debug("Visualization container added to layout.")

        # Effect controls
        logger.debug("Creating effect controls panel.")
        effect_panel: QtWidgets.QTabWidget = (
            QtWidgets.QTabWidget()
        )  # Create tab widget for effects, a tab widget to organize effect controls.
        logger.debug("Effect panel created.")
        self._create_effect_controls(
            effect_panel
        )  # Create effect control widgets, calls the method to add effect control widgets to the panel.
        logger.debug("Effect control widgets created.")
        layout.addWidget(
            effect_panel
        )  # Add effect panel to layout, adds the effect panel to the main layout.
        logger.debug("Effect panel added to layout.")
        logger.debug("UI initialization complete.")

    def _create_toolbar(self, toolbar: QtWidgets.QToolBar) -> None:
        """
        Creates the toolbar buttons.
        This method adds record, play, stop, load, and save buttons to the toolbar.

        Args:
            toolbar (QtWidgets.QToolBar): The toolbar to add buttons to.
        """
        logger.debug("Creating toolbar buttons: Starting button setup.")
        self.record_btn: QtWidgets.QPushButton = QtWidgets.QPushButton(
            "● Record"
        )  # Record button, a button to start/stop recording.
        logger.debug("Record button created.")
        self.play_btn: QtWidgets.QPushButton = QtWidgets.QPushButton(
            "▶ Play"
        )  # Play button, a button to play the audio.
        logger.debug("Play button created.")
        self.stop_btn: QtWidgets.QPushButton = QtWidgets.QPushButton(
            "◼ Stop"
        )  # Stop button, a button to stop audio playback.
        logger.debug("Stop button created.")
        self.load_btn: QtWidgets.QPushButton = QtWidgets.QPushButton(
            "📂 Load"
        )  # Load button, a button to load an audio file.
        logger.debug("Load button created.")
        self.save_btn: QtWidgets.QPushButton = QtWidgets.QPushButton(
            "💾 Save"
        )  # Save button, a button to save the processed audio.
        logger.debug("Save button created.")

        for btn in [
            self.record_btn,
            self.play_btn,
            self.stop_btn,
            self.load_btn,
            self.save_btn,
        ]:  # Add buttons to toolbar, iterates through the buttons and adds them to the toolbar.
            toolbar.addWidget(btn)
            logger.debug(f"Button {btn.text()} added to toolbar.")

        self.record_btn.clicked.connect(
            self.toggle_recording
        )  # Connect record button, connects the record button's click signal to the toggle_recording method.
        logger.debug("Record button connected to toggle_recording.")
        self.play_btn.clicked.connect(
            self.play_audio
        )  # Connect play button, connects the play button's click signal to the play_audio method.
        logger.debug("Play button connected to play_audio.")
        self.stop_btn.clicked.connect(
            self.stop_audio
        )  # Connect stop button, connects the stop button's click signal to the stop_audio method.
        logger.debug("Stop button connected to stop_audio.")
        self.load_btn.clicked.connect(
            self.load_audio_file
        )  # Connect load button, connects the load button's click signal to the load_audio_file method.
        logger.debug("Load button connected to load_audio_file.")
        self.save_btn.clicked.connect(
            self.save_processed_audio
        )  # Connect save button, connects the save button's click signal to the save_processed_audio method.
        logger.debug("Save button connected to save_processed_audio.")
        logger.debug("Toolbar buttons setup complete.")

    def _create_visualizations(self, container: QtWidgets.QSplitter) -> None:
        """
        Creates the visualization widgets.
        This method sets up the waveform, spectrum, and meter plot widgets.

        Args:
            container (QtWidgets.QSplitter): The container to add visualization widgets to.
        """
        logger.debug("Creating visualization widgets: Starting widget setup.")
        # Waveform plot
        logger.debug("Setting up waveform plot.")
        self.waveform.setLabel(
            "left", "Amplitude"
        )  # Set y-axis label, sets the label for the y-axis of the waveform plot.
        logger.debug("Waveform y-axis label set to 'Amplitude'.")
        self.waveform.setLabel(
            "bottom", "Time (s)"
        )  # Set x-axis label, sets the label for the x-axis of the waveform plot.
        logger.debug("Waveform x-axis label set to 'Time (s)'.")
        self.waveform.setYRange(
            -1, 1
        )  # Set y-axis range, sets the range of the y-axis of the waveform plot.
        logger.debug("Waveform y-axis range set to (-1, 1).")
        self.waveform.disableAutoRange()  # Prevent auto-range recursion, disables auto-ranging for the waveform plot.
        logger.debug("Waveform auto-range disabled.")

        # Meter plot
        logger.debug("Setting up meter plot.")
        self.meter.setLabel(
            "bottom", "RMS Level"
        )  # Set x-axis label, sets the label for the x-axis of the meter plot.
        logger.debug("Meter x-axis label set to 'RMS Level'.")
        self.meter.setXRange(
            0, 1
        )  # Set x-axis range, sets the range of the x-axis of the meter plot.
        logger.debug("Meter x-axis range set to (0, 1).")
        self.meter.setYRange(
            0, 1
        )  # Set y-axis range, sets the range of the y-axis of the meter plot.
        logger.debug("Meter y-axis range set to (0, 1).")
        self.meter.hideAxis("left")  # Hide y-axis, hides the y-axis of the meter plot.
        logger.debug("Meter y-axis hidden.")
        self.meter.disableAutoRange()  # Prevent meter auto-range issues, disables auto-ranging for the meter plot.
        logger.debug("Meter auto-range disabled.")

        container.addWidget(
            self.waveform
        )  # Add waveform plot to container, adds the waveform plot to the visualization container.
        logger.debug("Waveform plot added to container.")
        container.addWidget(
            self.spectrum
        )  # Add spectrum analyzer to container, adds the spectrum analyzer to the visualization container.
        logger.debug("Spectrum analyzer added to container.")
        container.addWidget(
            self.meter
        )  # Add meter plot to container, adds the meter plot to the visualization container.
        logger.debug("Meter plot added to container.")
        logger.debug("Visualization widgets setup complete.")

    def _create_effect_controls(self, panel: QtWidgets.QTabWidget) -> None:
        """
        Creates the effect control widgets.
        This method sets up the EQ, dynamics, noise reduction, speed, and pitch control tabs.

        Args:
            panel (QtWidgets.QTabWidget): The tab widget to add effect controls to.
        """
        logger.debug("Creating effect control widgets: Starting widget setup.")
        # EQ Controls
        logger.debug("Setting up EQ controls.")
        eq_tab: QtWidgets.QWidget = (
            QtWidgets.QWidget()
        )  # Create EQ tab, a widget for EQ controls.
        logger.debug("EQ tab created.")
        eq_layout: QtWidgets.QFormLayout = QtWidgets.QFormLayout(
            eq_tab
        )  # Create layout for EQ tab, a form layout for the EQ tab.
        logger.debug("EQ layout created.")
        self.low_shelf_slider: QtWidgets.QSlider = self._create_slider(
            -12, 12
        )  # Low shelf slider, a slider for adjusting the low shelf gain.
        logger.debug("Low shelf slider created.")
        self.mid_boost_slider: QtWidgets.QSlider = self._create_slider(
            -12, 12
        )  # Mid boost slider, a slider for adjusting the mid boost gain.
        logger.debug("Mid boost slider created.")
        self.high_shelf_slider: QtWidgets.QSlider = self._create_slider(
            -12, 12
        )  # High shelf slider, a slider for adjusting the high shelf gain.
        logger.debug("High shelf slider created.")
        eq_layout.addRow(
            "Low Shelf (200Hz)", self.low_shelf_slider
        )  # Add low shelf slider to layout, adds the low shelf slider to the EQ layout.
        logger.debug("Low shelf slider added to layout.")
        eq_layout.addRow(
            "Mid Boost (1-4kHz)", self.mid_boost_slider
        )  # Add mid boost slider to layout, adds the mid boost slider to the EQ layout.
        logger.debug("Mid boost slider added to layout.")
        eq_layout.addRow(
            "High Shelf (5kHz)", self.high_shelf_slider
        )  # Add high shelf slider to layout, adds the high shelf slider to the EQ layout.
        logger.debug("High shelf slider added to layout.")

        # Dynamics Controls
        logger.debug("Setting up dynamics controls.")
        dyn_tab: QtWidgets.QWidget = (
            QtWidgets.QWidget()
        )  # Create dynamics tab, a widget for dynamics controls.
        logger.debug("Dynamics tab created.")
        dyn_layout: QtWidgets.QFormLayout = QtWidgets.QFormLayout(
            dyn_tab
        )  # Create layout for dynamics tab, a form layout for the dynamics tab.
        logger.debug("Dynamics layout created.")
        self.threshold_slider: QtWidgets.QSlider = self._create_slider(
            -40, 0, -20
        )  # Threshold slider, a slider for adjusting the compressor threshold.
        logger.debug("Threshold slider created.")
        self.ratio_slider: QtWidgets.QSlider = self._create_slider(
            1, 10, 4
        )  # Ratio slider, a slider for adjusting the compressor ratio.
        logger.debug("Ratio slider created.")
        self.attack_slider: QtWidgets.QSlider = self._create_slider(
            1, 100, 10
        )  # Attack slider, a slider for adjusting the compressor attack time.
        logger.debug("Attack slider created.")
        self.release_slider: QtWidgets.QSlider = self._create_slider(
            10, 500, 100
        )  # Release slider, a slider for adjusting the compressor release time.
        logger.debug("Release slider created.")
        dyn_layout.addRow(
            "Threshold (dB)", self.threshold_slider
        )  # Add threshold slider to layout, adds the threshold slider to the dynamics layout.
        logger.debug("Threshold slider added to layout.")
        dyn_layout.addRow(
            "Ratio", self.ratio_slider
        )  # Add ratio slider to layout, adds the ratio slider to the dynamics layout.
        logger.debug("Ratio slider added to layout.")
        dyn_layout.addRow(
            "Attack (ms)", self.attack_slider
        )  # Add attack slider to layout, adds the attack slider to the dynamics layout.
        logger.debug("Attack slider added to layout.")
        dyn_layout.addRow(
            "Release (ms)", self.release_slider
        )  # Add release slider to layout, adds the release slider to the dynamics layout.
        logger.debug("Release slider added to layout.")

        # Noise Reduction
        logger.debug("Setting up noise reduction controls.")
        nr_tab: QtWidgets.QWidget = (
            QtWidgets.QWidget()
        )  # Create noise reduction tab, a widget for noise reduction controls.
        logger.debug("Noise reduction tab created.")
        nr_layout: QtWidgets.QFormLayout = QtWidgets.QFormLayout(
            nr_tab
        )  # Create layout for noise reduction tab, a form layout for the noise reduction tab.
        logger.debug("Noise reduction layout created.")
        self.reduction_slider: QtWidgets.QSlider = self._create_slider(
            0, 40, 20
        )  # Reduction slider, a slider for adjusting the noise reduction amount.
        logger.debug("Reduction slider created.")
        nr_layout.addRow(
            "Reduction (dB)", self.reduction_slider
        )  # Add reduction slider to layout, adds the reduction slider to the noise reduction layout.
        logger.debug("Reduction slider added to layout.")

        # Store checkbox references
        logger.debug("Creating effect enable checkboxes.")
        self.eq_check: QtWidgets.QCheckBox = QtWidgets.QCheckBox(
            "Enable EQ"
        )  # EQ enable checkbox, a checkbox to enable/disable the EQ effect.
        logger.debug("EQ enable checkbox created.")
        self.dyn_check: QtWidgets.QCheckBox = QtWidgets.QCheckBox(
            "Enable Dynamics"
        )  # Dynamics enable checkbox, a checkbox to enable/disable the dynamics effect.
        logger.debug("Dynamics enable checkbox created.")
        self.nr_check: QtWidgets.QCheckBox = QtWidgets.QCheckBox(
            "Enable Noise Reduction"
        )  # Noise reduction enable checkbox, a checkbox to enable/disable the noise reduction effect.
        logger.debug("Noise reduction enable checkbox created.")

        for tab, check in [
            (eq_tab, self.eq_check),
            (dyn_tab, self.dyn_check),
            (nr_tab, self.nr_check),
        ]:  # Add checkboxes to tabs, iterates through the tabs and adds the corresponding checkboxes.
            check.setChecked(
                True
            )  # Set checkbox to checked, sets the checkbox to be initially checked.
            logger.debug(f"Checkbox {check.text()} set to checked.")
            tab.layout().addRow(
                check
            )  # Add checkbox to layout, adds the checkbox to the layout of the current tab.
            logger.debug(f"Checkbox {check.text()} added to layout.")
            check.stateChanged.connect(
                self._update_effect_states
            )  # Connect checkbox to update effect states, connects the checkbox's state change signal to the _update_effect_states method.
            logger.debug(f"Checkbox {check.text()} connected to _update_effect_states.")

        panel.addTab(
            eq_tab, "EQ"
        )  # Add EQ tab to panel, adds the EQ tab to the tab widget.
        logger.debug("EQ tab added to panel.")
        panel.addTab(
            dyn_tab, "Dynamics"
        )  # Add dynamics tab to panel, adds the dynamics tab to the tab widget.
        logger.debug("Dynamics tab added to panel.")
        panel.addTab(
            nr_tab, "Noise Reduction"
        )  # Add noise reduction tab to panel, adds the noise reduction tab to the tab widget.
        logger.debug("Noise reduction tab added to panel.")

        # Connect sliders to effect chain update
        logger.debug("Connecting sliders to effect chain update.")
        self.low_shelf_slider.valueChanged.connect(
            self._update_effect_chain
        )  # Connect low shelf slider, connects the low shelf slider's value change signal to the _update_effect_chain method.
        logger.debug("Low shelf slider connected to _update_effect_chain.")
        self.mid_boost_slider.valueChanged.connect(
            self._update_effect_chain
        )  # Connect mid boost slider, connects the mid boost slider's value change signal to the _update_effect_chain method.
        logger.debug("Mid boost slider connected to _update_effect_chain.")
        self.high_shelf_slider.valueChanged.connect(
            self._update_effect_chain
        )  # Connect high shelf slider, connects the high shelf slider's value change signal to the _update_effect_chain method.
        logger.debug("High shelf slider connected to _update_effect_chain.")
        self.threshold_slider.valueChanged.connect(
            self._update_effect_chain
        )  # Connect threshold slider, connects the threshold slider's value change signal to the _update_effect_chain method.
        logger.debug("Threshold slider connected to _update_effect_chain.")
        self.ratio_slider.valueChanged.connect(
            self._update_effect_chain
        )  # Connect ratio slider, connects the ratio slider's value change signal to the _update_effect_chain method.
        logger.debug("Ratio slider connected to _update_effect_chain.")
        self.attack_slider.valueChanged.connect(
            self._update_effect_chain
        )  # Connect attack slider, connects the attack slider's value change signal to the _update_effect_chain method.
        logger.debug("Attack slider connected to _update_effect_chain.")
        self.release_slider.valueChanged.connect(
            self._update_effect_chain
        )  # Connect release slider, connects the release slider's value change signal to the _update_effect_chain method.
        logger.debug("Release slider connected to _update_effect_chain.")
        self.reduction_slider.valueChanged.connect(
            self._update_effect_chain
        )  # Connect reduction slider, connects the reduction slider's value change signal to the _update_effect_chain method.
        logger.debug("Reduction slider connected to _update_effect_chain.")

        # Add speed control tab
        logger.debug("Setting up speed control tab.")
        speed_tab: QtWidgets.QWidget = (
            QtWidgets.QWidget()
        )  # Create speed tab, a widget for speed controls.
        logger.debug("Speed tab created.")
        speed_layout: QtWidgets.QFormLayout = QtWidgets.QFormLayout(
            speed_tab
        )  # Create layout for speed tab, a form layout for the speed tab.
        logger.debug("Speed layout created.")
        self.speed_slider: QtWidgets.QSlider = self._create_slider(
            50, 200, 100
        )  # Speed slider, a slider for adjusting the playback speed.
        logger.debug("Speed slider created.")
        speed_layout.addRow(
            "Speed (%)", self.speed_slider
        )  # Add speed slider to layout, adds the speed slider to the speed layout.
        logger.debug("Speed slider added to layout.")
        panel.addTab(
            speed_tab, "Speed"
        )  # Add speed tab to panel, adds the speed tab to the tab widget.
        logger.debug("Speed tab added to panel.")
        self.speed_slider.valueChanged.connect(
            self._update_effect_chain
        )  # Connect speed slider, connects the speed slider's value change signal to the _update_effect_chain method.
        logger.debug("Speed slider connected to _update_effect_chain.")

        # Add to existing effects
        logger.debug("Setting up pitch control tab.")
        pitch_tab: QtWidgets.QWidget = (
            QtWidgets.QWidget()
        )  # Create pitch tab, a widget for pitch controls.
        logger.debug("Pitch tab created.")
        pitch_layout: QtWidgets.QFormLayout = QtWidgets.QFormLayout(
            pitch_tab
        )  # Create layout for pitch tab, a form layout for the pitch tab.
        logger.debug("Pitch layout created.")
        self.pitch_slider: QtWidgets.QSlider = self._create_slider(
            -12, 12, 0
        )  # Pitch slider, a slider for adjusting the pitch shift.
        logger.debug("Pitch slider created.")
        pitch_layout.addRow(
            "Pitch Shift (semitones)", self.pitch_slider
        )  # Add pitch slider to layout, adds the pitch slider to the pitch layout.
        logger.debug("Pitch slider added to layout.")
        panel.addTab(
            pitch_tab, "Pitch"
        )  # Add pitch tab to panel, adds the pitch tab to the tab widget.
        logger.debug("Pitch tab added to panel.")

        # Connect new slider
        self.pitch_slider.valueChanged.connect(
            self._update_effect_chain
        )  # Connect pitch slider, connects the pitch slider's value change signal to the _update_effect_chain method.
        logger.debug("Pitch slider connected to _update_effect_chain.")
        logger.debug("Effect control widgets setup complete.")

    def _create_slider(
        self, min_val: int, max_val: int, default: int = 0
    ) -> QtWidgets.QSlider:
        """
        Creates a QSlider with specified range and default value.

        Args:
            min_val (int): Minimum value of the slider.
            max_val (int): Maximum value of the slider.
            default (int, optional): Default value of the slider. Defaults to 0.

        Returns:
            QtWidgets.QSlider: The created slider.
        """
        logger.debug(
            f"Creating slider with min={min_val}, max={max_val}, default={default}."
        )
        slider = QtWidgets.QSlider(
            QtCore.Qt.Orientation.Horizontal
        )  # Create a horizontal slider.
        logger.debug("QSlider created.")
        slider.setRange(min_val, max_val)  # Set the range of the slider.
        logger.debug(f"Slider range set to ({min_val}, {max_val}).")
        slider.setValue(default)  # Set the default value of the slider.
        logger.debug(f"Slider default value set to {default}.")
        return slider

    def _audio_callback(self, indata: np.ndarray, frames: int, time, status) -> None:
        """
        Callback function for the audio stream.
        This method is called whenever new audio data is available from the input stream.

        Args:
            indata (np.ndarray): The incoming audio data as a NumPy array.
            frames (int): The number of frames in the audio data.
            time: The timestamp of the audio data.
            status: The status of the audio stream.
        """
        with self.audio_lock:  # Acquire the audio lock to ensure thread-safe access to audio data.
            logger.debug("Audio callback: Audio lock acquired.")
            if self.is_recording:  # Check if recording is active.
                logger.debug("Audio callback: Recording is active.")
                self.recording_buffer.append(
                    indata.copy()
                )  # Append the incoming audio data to the recording buffer.
                logger.debug(
                    "Audio callback: Incoming data appended to recording buffer."
                )
                if (
                    len(indata) > 1024
                ):  # Check if the incoming data is large enough to process.
                    logger.debug("Audio callback: Processing audio data.")
                    self.processed_data = self.engine.process_audio(
                        indata
                    )  # Process the audio data using the audio engine.
                    logger.debug("Audio callback: Audio data processed.")
                    self.update_signal.emit()  # Emit the update signal to refresh the UI.
                    logger.debug("Audio callback: Update signal emitted.")
            else:
                logger.debug("Audio callback: Recording is not active.")
        logger.debug("Audio callback: Audio lock released.")

    def toggle_recording(self) -> None:
        """
        Toggles the recording state.
        This method starts or stops the audio stream and updates the UI accordingly.
        """
        logger.info(
            f"Toggling recording: Current state is_recording={self.is_recording}."
        )
        self.is_recording = not self.is_recording  # Toggle the recording state.
        logger.debug(f"Recording state toggled to {self.is_recording}.")
        if self.is_recording:  # If recording is starting.
            logger.info("Starting recording.")
            self.recording_buffer.clear()  # Clear the recording buffer.
            logger.debug("Recording buffer cleared.")
            self.stream = sd.InputStream(  # Recreate stream each time, creates a new audio input stream.
                samplerate=AUDIO_CONFIG["sample_rate"],
                channels=AUDIO_CONFIG["channels"],
                dtype=AUDIO_CONFIG["dtype"],
                blocksize=AUDIO_CONFIG["blocksize"],
                callback=self._audio_callback,
            )
            logger.debug("New audio stream created.")
            self.stream.start()  # Start the audio stream.
            logger.debug("Audio stream started.")
            self.record_btn.setText(
                "⏹ Stop"
            )  # Update the record button text to "Stop".
            logger.debug("Record button text updated to '⏹ Stop'.")
        else:  # If recording is stopping.
            logger.info("Stopping recording.")
            self.stream.stop()  # Stop the audio stream.
            logger.debug("Audio stream stopped.")
            self.stream.close()  # Close the audio stream.
            logger.debug("Audio stream closed.")
            QtCore.QThread.msleep(100)  # Allow stream to fully stop
            self.audio_data = np.concatenate(
                list(self.recording_buffer)
            )  # Concatenate the recording buffer into a single array.
            logger.debug("Recording buffer concatenated into audio data.")
            self.processed_data = self.engine.process_audio(
                self.audio_data
            )  # Process the audio data using the audio engine.
            logger.debug("Audio data processed.")
            self.record_btn.setText(
                "● Record"
            )  # Update the record button text to "Record".
            logger.debug("Record button text updated to '● Record'.")
            self._update_waveform()  # Update the waveform display.
            logger.debug("Waveform updated.")

    def play_audio(self):
        """Handle audio playback with current processing"""
        if self.processed_data is not None:
            logger.info("Playing processed audio")
            sd.play(self.processed_data, self.sample_rate)
        elif self.audio_data is not None:
            logger.info("Playing raw audio")
            sd.play(self.audio_data, self.sample_rate)
        else:
            logger.warning("No audio to play")

    def stop_audio(self):
        """Stop all audio playback"""
        logger.info("Stopping audio playback")
        sd.stop()
        self.timer.stop()

    def load_audio_file(self):
        """Load audio file with professional format support"""
        file_dialog = QtWidgets.QFileDialog(self)
        file_dialog.setNameFilters(
            ["Audio Files (*.wav *.flac *.aiff *.ogg)", "All Files (*)"]
        )

        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            try:
                self.raw_audio_segment = AudioSegment.from_file(file_path)
                self.sample_rate = self.raw_audio_segment.frame_rate

                # Convert to numpy array
                raw_mono = self.raw_audio_segment.set_channels(1)
                self.audio_data = np.array(raw_mono.get_array_of_samples()).astype(
                    np.float32
                )
                self.audio_data /= np.iinfo(np.int16).max

                # Initialize processing chain
                self.processed_data = self.engine.process_audio(self.audio_data)
                self.proc_audio_segment = self._numpy_to_audiosegment(
                    self.processed_data, self.sample_rate
                )
                self._update_waveform()
                self.engine.sample_rate = self.sample_rate
                logger.info(f"Loaded {file_path}")

            except Exception as e:
                logger.error(f"Load error: {str(e)}")

    def save_processed_audio(self):
        """Save processed audio in broadcast WAV format"""
        if self.proc_audio_segment is None:
            logger.warning("No processed audio to save")
            return

        file_dialog = QtWidgets.QFileDialog(self)
        file_dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        file_dialog.setNameFilter("Wave Audio (*.wav)")

        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            if not file_path.lower().endswith(".wav"):
                file_path += ".wav"

            # Export as 24-bit WAV
            self.proc_audio_segment.export(
                file_path, format="wav", parameters=["-sample_fmt", "s32"]
            )
            logger.info(f"Saved processed audio to {file_path}")

    def _update_displays(self):
        """Update all visualizations in real-time"""
        display_data = (
            self.processed_data if self.processed_data is not None else self.audio_data
        )
        if display_data is not None:
            # Show last 2 seconds of audio for real-time monitoring
            samples = int(2 * self.sample_rate)
            recent_data = (
                display_data[-samples:] if len(display_data) > samples else display_data
            )

            # Waveform display
            self.waveform.clear()
            time_axis = np.linspace(
                0, len(recent_data) / self.sample_rate, len(recent_data)
            )
            self.waveform.plot(time_axis, recent_data.squeeze(), pen="c")

            # Spectrum analysis
            self.spectrum.update_spectrum(recent_data, self.sample_rate)

            # RMS metering in dBFS
            rms = np.sqrt(np.mean(recent_data**2))
            dbfs = 20 * np.log10(rms + 1e-9)
            self.meter.clear()
            self.meter.plot([0, 1], [0, 0], brush=pg.mkBrush((0, 255, 0, 100)))
            self.meter.plot(
                [0, np.clip(-dbfs / 90, 0, 1)],
                [0, 0],
                brush=pg.mkBrush((255, 0, 0, 150)),
                pen=None,
            )

    def _update_waveform(self):
        """Update waveform display after processing"""
        if self.processed_data is not None:
            self.waveform.clear()
            time_axis = np.linspace(
                0, len(self.processed_data) / self.sample_rate, len(self.processed_data)
            )
            self.waveform.plot(time_axis, self.processed_data.squeeze(), pen="y")

    def _numpy_to_audiosegment(self, data, sample_rate):
        """Convert processed numpy array back to AudioSegment"""
        # Proper 32-bit PCM conversion
        int_data = np.clip(data * 2147483647, -2147483648, 2147483647).astype(np.int32)
        byte_data = int_data.tobytes()

        with io.BytesIO() as wav_buffer:
            with wave.open(wav_buffer, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(4)  # 32-bit
                wf.setframerate(sample_rate)
                wf.writeframes(byte_data)

            wav_buffer.seek(0)
            return AudioSegment.from_file(wav_buffer, format="wav")

    def _update_effect_states(self):
        self.active_effects["eq"] = self.eq_check.isChecked()
        self.active_effects["compressor"] = self.dyn_check.isChecked()
        self.active_effects["denoise"] = self.nr_check.isChecked()
        self._update_effect_chain()

    def _update_effect_chain(self):
        """Update the effect chain with current parameters"""
        self.engine.effect_chain = []

        # Order matters: noise reduction first
        if self.active_effects["denoise"]:
            self.engine.add_effect(
                "denoise", {"reduction_db": self.reduction_slider.value()}
            )

        if self.active_effects["eq"]:
            self.engine.add_effect(
                "eq",
                {
                    "low_shelf": self.low_shelf_slider.value(),
                    "mid_boost": self.mid_boost_slider.value(),
                    "high_shelf": self.high_shelf_slider.value(),
                },
            )

        if self.active_effects["compressor"]:
            self.engine.add_effect(
                "compressor",
                {
                    "threshold": self.threshold_slider.value(),
                    "ratio": self.ratio_slider.value(),
                    "attack": self.attack_slider.value() / 1000,
                    "release": self.release_slider.value() / 1000,
                },
            )

        # Time-based effects last
        speed_factor = self.speed_slider.value() / 100
        if speed_factor != 1.0:
            self.engine.add_effect("speed", {"factor": speed_factor})

        if self.pitch_slider.value() != 0:
            self.engine.add_effect(
                "pitch", {"factor": 2 ** (self.pitch_slider.value() / 12)}
            )

        # Force reprocess when parameters change
        if self.audio_data is not None:
            self.processed_data = self.engine.process_audio(self.audio_data)
            self._update_waveform()

    def _init_undo_redo(self):
        undo_action = QtWidgets.QAction("Undo", self)
        undo_action.setShortcut("Ctrl+Z")
        undo_action.triggered.connect(self.undo)

        redo_action = QtWidgets.QAction("Redo", self)
        redo_action.setShortcut("Ctrl+Shift+Z")
        redo_action.triggered.connect(self.redo)

        self.addAction(undo_action)
        self.addAction(redo_action)

    def save_state(self):
        self.history.push_state(
            {
                "audio": (
                    self.audio_data.copy() if self.audio_data is not None else None
                ),
                "processed": (
                    self.processed_data.copy()
                    if self.processed_data is not None
                    else None
                ),
                "effects": [e.copy() for e in self.engine.effect_chain],
            }
        )

    def undo(self):
        if self.history.position > 0:
            self.history.position -= 1
            self._restore_state()

    def redo(self):
        if self.history.position < len(self.history.stack) - 1:
            self.history.position += 1
            self._restore_state()

    def _restore_state(self):
        state = self.history.stack[self.history.position]
        self.audio_data = state["audio"].copy() if state["audio"] is not None else None
        self.processed_data = (
            state["processed"].copy() if state["processed"] is not None else None
        )
        self.engine.effect_chain = [e.copy() for e in state["effects"]]
        self._update_waveform()

    def closeEvent(self, event):
        """Clean up resources on window close"""
        self.stream.close()
        sd.stop()
        self.timer.stop()
        event.accept()


def main():
    """Application entry point"""
    # Force XCB platform for Linux compatibility
    os.environ["QT_QPA_PLATFORM"] = "xcb"

    app = QtWidgets.QApplication(sys.argv)
    editor = ProfessionalVoiceEditor()
    editor.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
