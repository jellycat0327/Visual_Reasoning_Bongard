import math
import random
import numpy as np
from PIL import Image, ImageFilter, ImageDraw


from TileHandler import rand_uniform, rotate

class BPGenerator:


    def __init__(self, img_size=(100, 100)):

        self.img_size = img_size

        return

    # use this when you want to convert an image to a format to be input to a CNN model
    def vectorize_img(self, img):
        dim = img.size[0]
        pix = np.array(img)/255.
        img_vec = pix.reshape(dim, dim, 1)
        return np.array(img_vec, dtype=np.float16)
    
    # use this when you want to convert an image to a flattened vector
    # (for example when you want to run logistic regression on the raw pixel values
    def flatten_img(self, img):
        pix = np.array(img)/255.
        return np.array(pix.flatten(), dtype=np.float16)


    def draw_tile(self, problem_number, side):

        if problem_number == 61:

            return draw_BP_61(side, self.img_size)



def draw_BP_61(side, img_size):

    # same number of crosses on each side of line, vs. different number

    img_width, img_height = img_size

    img = Image.new('L', img_size, "white")

    img_draw = ImageDraw.Draw(img, 'L')

    '''
    use [0, 1]x[0, 1] coordinates initially
    +y
    ^
    |
    +---> +x
    '''

    center = (0.5, 0.5)

    # create a horizontal line
    line_points = [(rand_uniform(0.1, 0.4), 0.5), (rand_uniform(0.6, 0.9), 0.5)]

    # create crosses above and below line
    cross_points = []

    top_num = 0
    bot_num = 0

    num_choices = list(range(6))
    random.shuffle(num_choices)

    if side == "left":
        # same number on each side
        top_num = num_choices[0]
        bot_num = num_choices[0]

    elif side == "right":
        # different number on each side
        top_num = num_choices[0]
        bot_num = num_choices[1]



    dy = 0.45/5

    upper_y_choices = [0.5+k*dy for k in range(1, 6)]
    random.shuffle(upper_y_choices)

    lower_y_choices = [0.5-k*dy for k in range(1, 6)]
    random.shuffle(lower_y_choices)

    for _ in range(top_num):
        # add a point above the line
        random_angle = rand_uniform(-0.85*math.pi/2, 0.85*math.pi/2)
        cross_points.append(rotate(center, (0.5, upper_y_choices.pop()), random_angle))

    for _ in range(bot_num):
        # add a point below the line
        random_angle = rand_uniform(-0.85*math.pi/2, 0.85*math.pi/2)
        cross_points.append(rotate(center, (0.5, lower_y_choices.pop()), random_angle))

    # rotate everything by a random amount
    rotation = rand_uniform(0, 2*math.pi)

    line_points = [rotate(center, pt, rotation) for pt in line_points]
    cross_points = [rotate(center, pt, rotation) for pt in cross_points]

    # convert all points to image pixel coordinates
    line_points = [(int(x*img_width), int((1-y)*img_height)) for x, y in line_points]
    cross_points = [(int(x*img_width), int((1-y)*img_height)) for x, y in cross_points]

    img_draw.line(line_points, width=int(3.*img_width/100.))
    for center in cross_points:
        radius = int(3.*img_width/100.)

        top_left = (center[0] - radius, center[1] - radius)
        bot_right = (center[0] + radius, center[1] + radius)

        coords = [top_left, bot_right]

        img_draw.ellipse(coords, fill="black", outline="black")

    del img_draw

    return img



