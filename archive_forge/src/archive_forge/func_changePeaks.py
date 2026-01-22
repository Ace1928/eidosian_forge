import math
import itertools
import random
def changePeaks(self):
    """Order the peaks to change position, height, width and number."""
    if self.minpeaks is not None and self.maxpeaks is not None:
        npeaks = len(self.peaks_function)
        u = self.random.random()
        r = self.maxpeaks - self.minpeaks
        if u < 0.5:
            u = self.random.random()
            n = min(npeaks - self.minpeaks, int(round(r * u * self.number_severity)))
            for i in range(n):
                idx = self.random.randrange(len(self.peaks_function))
                self.peaks_function.pop(idx)
                self.peaks_position.pop(idx)
                self.peaks_height.pop(idx)
                self.peaks_width.pop(idx)
                self.last_change_vector.pop(idx)
        else:
            u = self.random.random()
            n = min(self.maxpeaks - npeaks, int(round(r * u * self.number_severity)))
            for i in range(n):
                self.peaks_function.append(self.random.choice(self.pfunc_pool))
                self.peaks_position.append([self.random.uniform(self.min_coord, self.max_coord) for _ in range(self.dim)])
                self.peaks_height.append(self.random.uniform(self.min_height, self.max_height))
                self.peaks_width.append(self.random.uniform(self.min_width, self.max_width))
                self.last_change_vector.append([self.random.random() - 0.5 for _ in range(self.dim)])
    for i in range(len(self.peaks_function)):
        shift = [self.random.random() - 0.5 for _ in range(len(self.peaks_position[i]))]
        shift_length = sum((s ** 2 for s in shift))
        shift_length = self.move_severity / math.sqrt(shift_length) if shift_length > 0 else 0
        shift = [shift_length * (1.0 - self.lambda_) * s + self.lambda_ * c for s, c in zip(shift, self.last_change_vector[i])]
        shift_length = sum((s ** 2 for s in shift))
        shift_length = self.move_severity / math.sqrt(shift_length) if shift_length > 0 else 0
        shift = [s * shift_length for s in shift]
        new_position = []
        final_shift = []
        for pp, s in zip(self.peaks_position[i], shift):
            new_coord = pp + s
            if new_coord < self.min_coord:
                new_position.append(2.0 * self.min_coord - pp - s)
                final_shift.append(-1.0 * s)
            elif new_coord > self.max_coord:
                new_position.append(2.0 * self.max_coord - pp - s)
                final_shift.append(-1.0 * s)
            else:
                new_position.append(new_coord)
                final_shift.append(s)
        self.peaks_position[i] = new_position
        self.last_change_vector[i] = final_shift
        change = self.random.gauss(0, 1) * self.height_severity
        new_value = change + self.peaks_height[i]
        if new_value < self.min_height:
            self.peaks_height[i] = 2.0 * self.min_height - self.peaks_height[i] - change
        elif new_value > self.max_height:
            self.peaks_height[i] = 2.0 * self.max_height - self.peaks_height[i] - change
        else:
            self.peaks_height[i] = new_value
        change = self.random.gauss(0, 1) * self.width_severity
        new_value = change + self.peaks_width[i]
        if new_value < self.min_width:
            self.peaks_width[i] = 2.0 * self.min_width - self.peaks_width[i] - change
        elif new_value > self.max_width:
            self.peaks_width[i] = 2.0 * self.max_width - self.peaks_width[i] - change
        else:
            self.peaks_width[i] = new_value
    self._optimum = None