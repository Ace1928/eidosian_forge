import json
import random
from dataclasses import dataclass
from functools import lru_cache
from math import ceil, floor, log
from typing import Dict, Iterator, List, Optional, Tuple
import mido
def token_to_midi_message(utils: VocabUtils, token: str, state: DecodeState, end_token_pause: float=3.0) -> Iterator[Tuple[Optional[mido.Message], DecodeState]]:
    if state is None:
        state = DecodeState(total_time=0.0, delta_accum=0.0, current_bin=utils.cfg._short_instrument_names_str_to_int[utils.cfg.short_instr_bin_names[0]], current_note=0, active_notes={})
    token = token.strip()
    if not token:
        yield (None, state)
        return
    if token == '<end>':
        d = end_token_pause * 1000.0
        state.delta_accum += d
        state.total_time += d
        if utils.cfg.decode_end_held_note_delay != 0.0:
            for (channel, note), start_time in list(state.active_notes.items()).copy():
                ticks = int(mido.second2tick(state.delta_accum / 1000.0, 480, 500000))
                state.delta_accum = 0.0
                del state.active_notes[channel, note]
                yield (mido.Message('note_off', note=note, time=ticks, channel=channel), state)
        yield (None, state)
        return
    if token.startswith('<'):
        yield (None, state)
        return
    if utils.cfg.unrolled_tokens:
        if token[0] == 't':
            d = utils.wait_token_to_delta(token)
            state.delta_accum += d
            state.total_time += d
        elif token[0] == 'n':
            state.current_note = int(token[1:], base=16)
        elif token[0] == 'i':
            state.current_bin = utils.cfg._short_instrument_names_str_to_int[token[1:]]
        elif token[0] == 'v':
            current_velocity = utils.bin_to_velocity(int(token[1:], base=16))
            channel = utils.cfg.bin_channel_map[utils.cfg.bin_instrument_names[state.current_bin]]
            ticks = int(mido.second2tick(state.delta_accum / 1000.0, 480, 500000))
            state.delta_accum = 0.0
            if current_velocity > 0:
                yield (mido.Message('note_on', note=state.current_note, velocity=current_velocity, time=ticks, channel=channel), state)
            else:
                yield (mido.Message('note_off', note=state.current_note, velocity=0, time=ticks, channel=channel), state)
    elif token[0] == 't' and token[1].isdigit():
        d = utils.wait_token_to_delta(token)
        state.delta_accum += d
        state.total_time += d
        if utils.cfg.decode_end_held_note_delay != 0.0:
            for (channel, note), start_time in list(state.active_notes.items()).copy():
                if state.total_time - start_time > utils.cfg.decode_end_held_note_delay * 1000.0:
                    ticks = int(mido.second2tick(state.delta_accum / 1000.0, 480, 500000))
                    state.delta_accum = 0.0
                    del state.active_notes[channel, note]
                    yield (mido.Message('note_off', note=note, time=ticks, channel=channel), state)
                    return
    else:
        bin, note, velocity = utils.note_token_to_data(token)
        channel = utils.cfg.bin_channel_map[utils.cfg.bin_instrument_names[bin]]
        ticks = int(mido.second2tick(state.delta_accum / 1000.0, 480, 500000))
        state.delta_accum = 0.0
        if velocity > 0:
            if utils.cfg.decode_fix_repeated_notes:
                if (channel, note) in state.active_notes:
                    del state.active_notes[channel, note]
                    yield (mido.Message('note_off', note=note, time=ticks, channel=channel), state)
                    ticks = 0
            state.active_notes[channel, note] = state.total_time
            yield (mido.Message('note_on', note=note, velocity=velocity, time=ticks, channel=channel), state)
            return
        else:
            if (channel, note) in state.active_notes:
                del state.active_notes[channel, note]
            yield (mido.Message('note_off', note=note, time=ticks, channel=channel), state)
            return
    yield (None, state)