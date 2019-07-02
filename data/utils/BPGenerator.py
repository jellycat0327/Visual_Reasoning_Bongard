import math
import random
import numpy as np
from PIL import Image, ImageFilter, ImageDraw


from utils.TileHandler import *
from utils.drawing_utils import *

class BPGenerator:


	def __init__(self, img_size=(100, 100)):

		self.img_size = img_size

		return

	def vectorize_img(self, img):
		dim = img.size[0]
		pix = np.array(img)/255.
		img_vec = pix.reshape(dim, dim, 1)
		return np.array(img_vec, dtype=np.float16)

	def flatten_img(self, img):
		pix = np.array(img)/255.
		return np.array(pix.flatten(), dtype=np.float16)


	def draw_tile(self, problem_number, side):

		if problem_number == "1":
			return draw_BP_1(side, self.img_size)

		elif problem_number == "2":
			return draw_BP_2(side, self.img_size)

		elif problem_number == "3":
			return draw_BP_3(side, self.img_size)

		elif problem_number == "5":
			return draw_BP_5(side, self.img_size)

		elif problem_number == "6":

			return draw_BP_6(side, self.img_size)

		elif problem_number == "7":

			return draw_BP_7(side, self.img_size)

		elif problem_number == "8":

			return draw_BP_8(side, self.img_size)

		elif problem_number == "13":

			return draw_BP_13(side, self.img_size)

		elif problem_number == "22":

			return draw_BP_22(side, self.img_size)

		elif problem_number == "61":

			return draw_BP_61(side, self.img_size)

		elif problem_number == "61-simple":

			return draw_BP_61_simple(side, self.img_size)


def draw_BP_1(side, img_size):

	# empty vs. not empty

	img_width, img_height = img_size

	img = Image.new('L', img_size, "white")

	if side == "left":
		return img

	#img_draw = ImageDraw.Draw(img, 'L')

	TH = TileHandler(n_shapes_range = (1, 1), size = img_width, line_k_range = None,
                 circle = True, dot = None, curve = None, polygon_k_range = (3, 8),
                 ellipses = True, equilateral_polygon_range = (3, 8))

	return TH.generate_tile()['img']

def draw_BP_2(side, img_size):

	# big vs. small

	img_width, img_height = img_size

	img = Image.new('L', img_size, "white")

	img_draw = ImageDraw.Draw(img, 'L')

	if side == "left":
		size_frac = rand_uniform(0.4, 0.8)
		filled = random.random() > 0.5

		if random.random() < 0.75:
			nb_sides = random.randint(3, 5)
			equilateral = True #random.random() > 0.5
			draw_polygon(img_draw, img, size_frac, filled, nb_sides, equilateral)
		else:
			draw_circle(img_draw, img, size_frac, filled)


	elif side == "right":
		size_frac = rand_uniform(0.2, 0.3)
		filled = random.random() > 0.5

		if random.random() < 0.75:
			nb_sides = random.randint(3, 5)
			equilateral = True #random.random() > 0.5
			draw_polygon(img_draw, img, size_frac, filled, nb_sides, equilateral)
		else:
			draw_circle(img_draw, img, size_frac, filled)
	
	del img_draw

	return img


def draw_BP_3(side, img_size):

	# outlined vs. filled

	img_width, img_height = img_size

	img = Image.new('L', img_size, "white")

	img_draw = ImageDraw.Draw(img, 'L')

	if side == "left":
		size_frac = rand_uniform(0.2, 0.8)
		filled = False

		if random.random() < 0.75:
			nb_sides = random.randint(3, 5)
			equilateral = random.random() > 0.5
			draw_polygon(img_draw, img, size_frac, filled, nb_sides, equilateral)
		else:
			draw_circle(img_draw, img, size_frac, filled)


	elif side == "right":
		size_frac = rand_uniform(0.2, 0.8)
		filled = True

		if random.random() < 0.75:
			nb_sides = random.randint(3, 5)
			equilateral = random.random() > 0.5
			draw_polygon(img_draw, img, size_frac, filled, nb_sides, equilateral)
		else:
			draw_circle(img_draw, img, size_frac, filled)
	
	del img_draw

	return img


def draw_BP_5(side, img_size):

	# polygon vs. round

	img_width, img_height = img_size

	img = Image.new('L', img_size, "white")

	if side == "left":

		TH = TileHandler(n_shapes_range = (1, 1), size = img_width, line_k_range = None,
        	         circle = None, dot = None, curve = None, polygon_k_range = (3, 5),
            	     ellipses = None, equilateral_polygon_range = (3,5))

		return TH.generate_tile()['img']

	elif side == "right":

		TH = TileHandler(n_shapes_range = (1, 1), size = img_width, line_k_range = None,
        	         circle = True, dot = None, curve = None, polygon_k_range = None,
            	     ellipses = True, equilateral_polygon_range = None)

		return TH.generate_tile()['img']

def draw_BP_6(side, img_size):

	# triangles vs. quadrilaterals

	img_width, img_height = img_size

	img = Image.new('L', img_size, "white")

	if side == "left":

		TH = TileHandler(n_shapes_range = (1, 1), size = img_width, line_k_range = None,
        	         circle = None, dot = None, curve = None, polygon_k_range = None,
            	     ellipses = None, equilateral_polygon_range = (3,3))

		return TH.generate_tile()['img']

	elif side == "right":

		TH = TileHandler(n_shapes_range = (1, 1), size = img_width, line_k_range = None,
        	         circle = None, dot = None, curve = None, polygon_k_range = None,
            	     ellipses = None, equilateral_polygon_range = (4,4))

		return TH.generate_tile()['img']

def draw_BP_7(side, img_size):

	# vertical things vs. horizontal things

	img_width, img_height = img_size

	img = Image.new('L', img_size, "white")

	img_draw = ImageDraw.Draw(img, 'L')

	width = rand_uniform(0.1, 0.2)
	height = rand_uniform(0.6, 0.8)

	center = (0.5, 0.5)

	top_left = (rand_uniform(0, 1-width), rand_uniform(height, 1))
	bot_right = (top_left[0]+width, top_left[1]-height)

	points = [top_left, bot_right]

	rotation = 0

	if side == "left":
		rotation = 0

	elif side == "right":
		rotation = math.pi/2


	points = [rotate(center, pt, rotation) for pt in points]

	points = [(int(x*img_width), int((1-y)*img_height)) for x, y in points]

	img_draw.rectangle(points, fill="black")
	
	del img_draw

	return img


def draw_BP_8(side, img_size):

	# on the right vs. on the left

	img_width, img_height = img_size

	img = Image.new('L', img_size, "white")

	img_draw = ImageDraw.Draw(img, 'L')

	width = rand_uniform(0.1, 0.4)
	height = rand_uniform(0.1, 0.4)

	center_x = 0.5

	if side == "left":
		# figure is on right side of tile
		top_left = (rand_uniform(0.5, 0.9), rand_uniform(0, 0.9))
		bot_right = (top_left[0] + rand_uniform(0.1, 1-top_left[0]), top_left[1] + rand_uniform(0.1, 1-top_left[1]))

	elif side == "right":
		top_left = (rand_uniform(0, 0.4), rand_uniform(0, 0.9))
		bot_right = (top_left[0] + rand_uniform(0.1, 0.5-top_left[0]), top_left[1] + rand_uniform(0.1, 1-top_left[1]))

	points = [top_left, bot_right]

	points = [(int(x*img_width), int((1-y)*img_height)) for x, y in points]

	img_draw.rectangle(points, fill="black")
	
	del img_draw

	return img


def draw_BP_13(side, img_size):

	# vertical rectangle or horizontal ellipse VS horizontal rectangle or vertical ellipse

	img_width, img_height = img_size

	img = Image.new('L', img_size, "white")

	img_draw = ImageDraw.Draw(img, 'L')


	width = rand_uniform(0.1, 0.2)
	height = rand_uniform(0.6, 0.8)

	center = (0.5, 0.5)

	top_left = (rand_uniform(0, 1-width), rand_uniform(height, 1))
	bot_right = (top_left[0]+width, top_left[1]-height)

	points = [top_left, bot_right]

	rotation = 0

	# rectangle
	if random.random() < 0.5:
		

		if side == "left":
			rotation = 0
		elif side == "right":
			rotation = math.pi/2


		points = [rotate(center, pt, rotation) for pt in points]

		points = [(int(x*img_width), int((1-y)*img_height)) for x, y in points]

		img_draw.rectangle(points, fill="black")
		
		del img_draw

		return img

	# ellipse
	else:

		if side == "left":
			rotation = math.pi/2
		elif side == "right":
			rotation = 0

		points = [rotate(center, pt, rotation) for pt in points]

		if side == "left":
			# need to mirror the points about the center horizontal
			points = [(pt[0], center[1] - (pt[1] - center[1])) for pt in points]

		points = [(int(x*img_width), int((1-y)*img_height)) for x, y in points]

		img = draw_ellipse(img, points, width=int(3.*img_height/100), outline='black', antialias=4)
		
		del img_draw

		return img


def draw_BP_22(side, img_size):

	# uniform areas of shapes (all small or all large) VS different areas of shapes (some small, some large)

	# TODO: currently, the shapes may overlap -- may want to prevent this

	img_width, img_height = img_size

	img = Image.new('L', img_size, "white")

	img_draw = ImageDraw.Draw(img, 'L')


	nb_figures = random.randint(2, 3)



	def draw_random_thing(size):

		#width = 0.1 if size == "small" else 0.3
		#height = 0.1 if size == "small" else 0.3

		'''
		top_left = (rand_uniform(0, 1-width), rand_uniform(height, 1))
		bot_right = (top_left[0]+width, top_left[1]-height)

		points = [top_left, bot_right]
		points = [(int(x*img_width), int((1-y)*img_height)) for x, y in points]

		img_draw.rectangle(points, fill="black")
		'''

		size_frac = 0.1 if size == "small" else 0.4
		filled = False

		if random.random() < 0.75:
			nb_sides = random.randint(3, 5)
			equilateral = True
			draw_polygon(img_draw, img, size_frac, filled, nb_sides, equilateral)
		else:
			draw_circle(img_draw, img, size_frac, filled)

		return


	if side == "left":
		# different sizes

		nb_small = random.randint(1, nb_figures-1)
		nb_big = nb_figures - nb_small

		for _ in range(nb_small):
			draw_random_thing("small")

		for _ in range(nb_big):
			draw_random_thing("big")

	elif side == "right":
		# same sizes
		size = random.choice(["big", "small"])
		for _ in range(nb_figures):
			draw_random_thing(size)

		



	del img_draw

	return img




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


def draw_BP_61_simple(side, img_size):

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

	num_choices = list(range(1, 3))
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
	rotation = 0 #rand_uniform(0, 2*math.pi)

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



if __name__ == "__main__":

	BPG = BPGenerator(img_size=(200, 200))
	img_l = BPG.draw_tile(problem_number="5", side="left")
	img_r = BPG.draw_tile(problem_number="5", side="right")