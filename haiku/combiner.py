from haiku import converter
from haiku import generator
from PIL import Image
import PIL.ImageOps
import numpy as np

PNG_CHAR_WIDTH = 28
PNG_CHAR_HEIGHT = 28

class Combiner:
    def __init__(self):
        self.converter = converter.Converter()
        self.generator = generator.Generator()
        self.make_new_haiku()

    def make_new_haiku(self):
        new_haiku = self.generator.write_haiku()
        uppercase_haiku = new_haiku.upper()
        print(uppercase_haiku)
        lines_np = []
        for line_str in uppercase_haiku.splitlines():
            lines_np.append(self.converter.convert(line_str))
        longest_line_length = max(len(l) for l in lines_np)
        print(longest_line_length)
        lines_np = [np.concatenate(line, axis=1) for line in lines_np]

        def pad_line(line):
            target = PNG_CHAR_WIDTH * longest_line_length
            to_add = np.zeros((PNG_CHAR_HEIGHT, target-line.shape[1]))
            return np.concatenate((line, to_add.astype(np.uint8)), axis=1)

        padded_lines = [pad_line(line) for line in lines_np]


        """for line in lines_np:
            while line.shape[1] < PNG_CHAR_WIDTH * longest_line_length:
                print(line.shape[1])
                line = np.concatenate((line, np.zeros((PNG_CHAR_WIDTH, PNG_CHAR_HEIGHT)).astype(np.uint8)), axis=1)
        for line in lines_np: print(line.shape)"""


        all_lines = np.concatenate(padded_lines, axis=0)
        im = Image.fromarray(all_lines, 'L')
        im = PIL.ImageOps.invert(im)
        return im, uppercase_haiku
