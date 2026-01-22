import pygame
import pygame.midi
from pygame.locals import *

# Initialize Pygame and the MIDI system
pygame.init()
pygame.midi.init()

# Set up the MIDI output
port = pygame.midi.get_default_output_id()
midi_out = pygame.midi.Output(port, 0)

# Define a simple mapping of keyboard keys to MIDI notes
# A4 is MIDI note 69, and we'll map more notes around it
key_to_note = {
    K_a: 60,  # C4
    K_s: 62,  # D4
    K_d: 64,  # E4
    K_f: 65,  # F4
    K_g: 67,  # G4
    K_h: 69,  # A4
    K_j: 71,  # B4
    K_k: 72,  # C5
}

# Modifier keys can shift the octave up or down
octave_shift = 0

# Set up the display
screen = pygame.display.set_mode((400, 300))
pygame.display.set_caption("Pygame MIDI Keyboard")

# Event loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        elif event.type == KEYDOWN:
            if event.key in key_to_note:
                # Check for modifiers
                if pygame.key.get_mods() & KMOD_SHIFT:
                    octave_shift = 12  # Shift up an octave
                elif pygame.key.get_mods() & KMOD_CTRL:
                    octave_shift = -12  # Shift down an octave
                else:
                    octave_shift = 0

                # Play the note
                note = key_to_note[event.key] + octave_shift
                midi_out.note_on(note, 127)  # 127 is the velocity
        elif event.type == KEYUP:
            if event.key in key_to_note:
                note = key_to_note[event.key] + octave_shift
                midi_out.note_off(note, 127)

# Close MIDI stream and Pygame
midi_out.close()
pygame.midi.quit()
pygame.quit()
