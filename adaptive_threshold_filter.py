import numpy as np
from numba import boolean, float32, int32, njit
from numba.experimental import jitclass

spec = [
    ('_block_size', int32),
    ('half_block_size', int32),
    ('_next_block_size', int32),
    ('increase_block', boolean),
    ('sd_threshold', float32),
    ('max_value', int32),
    ('_step_size', int32),
    ('offset', int32),
    ('max_block_size', int32),
]


@njit
def get_y12(y, half_block_size, num_rows):
    y1, y2 = y - half_block_size, y + half_block_size + 1  # + 1 to include the last

    if y1 < 0:
        y1 = 0  # top
    if y2 >= num_rows:
        y2 = num_rows - 1  # bottom
    return y1, y2


@njit
def get_x12(x, half_block_size, num_cols):
    # x1, x2 = x - half_block_size -1 , x + half_block_size
    x1, x2 = x - half_block_size, x + half_block_size + 1

    if x1 < 0:
        x1 = 0  # left
    if x2 >= num_cols:
        x2 = num_cols - 1  # right
    return x1, x2


@jitclass(spec)
class AdaptiveMeanThresholdFilter:
    def __init__(self, block_size,
                 increase_block: bool,
                 step_size=2,
                 sd_threshold=10.5,
                 max_value=255,
                 offset=0,
                 max_block_size=None
                 ):
        self.block_size = block_size
        self.half_block_size = block_size // 2
        self.step_size = step_size
        self.next_block_size = self.block_size

        self.increase_block = increase_block
        self.sd_threshold = sd_threshold
        self.max_value = max_value
        self.offset = offset
        if max_block_size is None:
            self.max_block_size = block_size * 2
        else:
            self.max_block_size = max_block_size

    @property
    def next_half_block_size(self):
        return self._next_block_size // 2

    @property
    def next_block_size(self):
        self._next_block_size += self.step_size
        return self._next_block_size

    @next_block_size.setter
    def next_block_size(self, value):
        self._next_block_size = value

    @property
    def block_size(self):
        return self._block_size

    @block_size.setter
    def block_size(self, value):
        if value % 2 != 1 or value <= 0:
            raise ValueError('block_size must be a odd number square')
        self._block_size = value

    @property
    def step_size(self):
        return self._step_size

    @step_size.setter
    def step_size(self, value):
        if value % 2 != 0:
            raise ValueError('step_size must be a even number')
        self._step_size = value

    def apply_filter(self, img):
        num_rows, num_cols = img.shape[:]
        output = np.empty_like(img)
        for row in range(num_rows):
            y1, y2 = get_y12(row, self.half_block_size, num_rows)

            for col in range(num_cols):
                x1, x2 = get_x12(col, self.half_block_size, num_cols)
                patch = img[y1:y2, x1:x2]

                while np.std(patch) < self.sd_threshold and self.increase_block and self.next_block_size <= self.max_block_size:
                    # probably the original kernal only include a huge black moduel(many consecutive modules)
                    new_y1, new_y2 = get_y12(
                        row, self.next_half_block_size, num_rows)
                    new_x1, new_x2 = get_x12(
                        col, self.next_half_block_size, num_cols)
                    patch = img[new_y1:new_y2, new_x1:new_x2]

                self.next_block_size = self.block_size  # reset the block size
                
                output[row, col] = 0 if img[row, col] <= np.round(
                    np.mean(patch)) - self.offset else self.max_value
        return output
