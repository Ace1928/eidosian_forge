from midi2audio import FluidSynth
import argparse
import os
import subprocess
def midi_to_audio(self, midi_file, audio_file):
    subprocess.call(['fluidsynth', '-ni', self.sound_font, midi_file, '-F', audio_file, '-r', str(self.sample_rate)])