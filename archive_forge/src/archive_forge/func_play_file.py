import argparse
import sys
import mido
from mido import Message, MidiFile, tempo2bpm
def play_file(output, filename, print_messages):
    midi_file = MidiFile(filename)
    print(f'Playing {midi_file.filename}.')
    length = midi_file.length
    print('Song length: {} minutes, {} seconds.'.format(int(length / 60), int(length % 60)))
    print('Tracks:')
    for i, track in enumerate(midi_file.tracks):
        print(f'  {i:2d}: {track.name.strip()!r}')
    for message in midi_file.play(meta_messages=True):
        if print_messages:
            sys.stdout.write(repr(message) + '\n')
            sys.stdout.flush()
        if isinstance(message, Message):
            output.send(message)
        elif message.type == 'set_tempo':
            print('Tempo changed to {:.1f} BPM.'.format(tempo2bpm(message.tempo)))
    print()