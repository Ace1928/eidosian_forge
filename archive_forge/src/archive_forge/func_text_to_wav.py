import io
import global_var
from fastapi import APIRouter, HTTPException, UploadFile, status
from starlette.responses import StreamingResponse
from pydantic import BaseModel
from utils.midi import *
from midi2audio import FluidSynth
@router.post('/text-to-wav', tags=['MIDI'])
def text_to_wav(body: TextToWavBody):
    """
    Install fluidsynth first, see more: https://github.com/FluidSynth/fluidsynth/wiki/Download#distributions
    """
    if global_var.get(global_var.Deploy_Mode) is True:
        raise HTTPException(status.HTTP_403_FORBIDDEN)
    text = body.text.strip()
    if not text.startswith('<start>'):
        text = '<start> ' + text
    if not text.endswith('<end>'):
        text = text + ' <end>'
    txt_path = f'midi/{body.wav_name}.txt'
    midi_path = f'midi/{body.wav_name}.mid'
    wav_path = f'midi/{body.wav_name}.wav'
    with open(txt_path, 'w') as f:
        f.write(text)
    txt_to_midi(TxtToMidiBody(txt_path=txt_path, midi_path=midi_path))
    midi_to_wav(MidiToWavBody(midi_path=midi_path, wav_path=wav_path, sound_font_path=body.sound_font_path))
    return 'success'