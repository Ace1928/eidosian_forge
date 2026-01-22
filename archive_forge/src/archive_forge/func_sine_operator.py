import math as _math
import struct as _struct
from random import uniform as _uniform
from pyglet.media.codecs.base import Source, AudioFormat, AudioData
def sine_operator(sample_rate=44800, frequency=440, index=1, modulator=None, envelope=None):
    """A sine wave generator that can be optionally modulated with another generator.

    This generator represents a single FM Operator. It can be used by itself as a
    simple sine wave, or modulated by another waveform generator. Multiple operators
    can be linked together in this way. For example::

        operator1 = sine_operator(samplerate=44800, frequency=1.22)
        operator2 = sine_operator(samplerate=44800, frequency=99, modulator=operator1)
        operator3 = sine_operator(samplerate=44800, frequency=333, modulator=operator2)
        operator4 = sine_operator(samplerate=44800, frequency=545, modulator=operator3)

    :Parameters:
        `sample_rate` : int
            Audio samples per second. (CD quality is 44100).
        `frequency` : float
            The frequency, in Hz, of the waveform you wish to generate.
        `index` : float
            The modulation index. Defaults to 1
        `modulator` : sine_operator
            An optional operator to modulate this one.
        `envelope` : :py:class:`pyglet.media.synthesis._Envelope`
            An optional Envelope to apply to the waveform.
    """
    envelope = envelope or FlatEnvelope(1).get_generator(sample_rate, duration=None)
    sin = _math.sin
    step = 2 * _math.pi * frequency / sample_rate
    i = 0
    if modulator:
        while True:
            yield (sin(i * step + index * next(modulator)) * next(envelope))
            i += 1
    else:
        while True:
            yield (sin(i * step) * next(envelope))
            i += 1