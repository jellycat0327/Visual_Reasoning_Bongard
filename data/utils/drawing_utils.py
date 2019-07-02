
import random
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
import math

def draw_polygon(img_draw, img, size_frac, filled, nb_sides, equilateral):
	# number of equilateral polygon sides
	k = nb_sides
	size= img.size[0]

	radius = size_frac/2 #rand_uniform(size_frac_range[0], size_frac_range[1])

	origin = (rand_uniform(radius, 1.-radius), rand_uniform(radius, 1.-radius))


	# Choose a random point on the circle and start rotating
	# (starting point chosen by rotating the top point by random angle)
	top_point = (origin[0], (origin[1] + radius))
	random_angle = rand_uniform(0, 2*math.pi)
	starting_point = rotate(origin, top_point, random_angle)
	
	coords = []

	if equilateral:

		coords = [rotate(origin, starting_point, i_p*2*math.pi/k) for i_p in range(k)]

	else:

		rotation_angles = np.random.uniform(1, 2, k)
		rotation_angles = rotation_angles / np.sum(rotation_angles)
		rotation_angles = [sum(rotation_angles[:i]) for i in range(0, len(rotation_angles))]

		coords = [rotate(origin, starting_point, angle*2*math.pi) for angle in rotation_angles]


	# convert coords to pixels
	coords = [(int(x*size), int(y*size)) for x, y in coords]

	if filled:
		img_draw.polygon(coords, fill="black", outline="black")

	else:
		coords.append(coords[0])
		img_draw.line(coords, width=int(3.*size/100))


def draw_circle(img_draw, img, size_frac, filled):
	size= img.size[0]

	radius = size_frac/2
	center = (rand_uniform(radius, 1.-radius), rand_uniform(radius, 1.-radius))

	top_left = (center[0] - radius, center[1] - radius)
	bot_right = (center[0] + radius, center[1] + radius)

	coords = [top_left, bot_right]

	# convert coords to pixels
	coords = [(int(x*size), int(y*size)) for x, y in coords]


	if filled:
		img_draw.ellipse(coords, fill="black", outline="black")
	else:
		img = draw_ellipse(img, coords, width = int(3.*size/100), outline='black', antialias=4)

def draw_ellipse(image, bounds, width=1, outline='white', antialias=4):
	# https://stackoverflow.com/questions/32504246/draw-ellipse-in-python-pil-with-line-thickness/34926008

	"""Improved ellipse drawing function, based on PIL.ImageDraw."""

	# Use a single channel image (mode='L') as mask.
	# The size of the mask can be increased relative to the imput image
	# to get smoother looking results. 
	mask = Image.new(
		size=[int(dim * antialias) for dim in image.size],
		mode='L', color='black')
	draw = ImageDraw.Draw(mask)

	# draw outer shape in white (color) and inner shape in black (transparent)
	for offset, fill in (width/-2.0, 'white'), (width/2.0, 'black'):
		left, top = [(value + offset) * antialias for value in bounds[0]]
		right, bottom = [(value - offset) * antialias for value in bounds[1]]
		draw.ellipse([left, top, right, bottom], fill=fill)

	# downsample the mask using PIL.Image.LANCZOS 
	# (a high-quality downsampling filter).
	mask = mask.resize(image.size, Image.LANCZOS)
	# paste outline color to input image through the mask
	image.paste(outline, mask=mask)
	return image

def rand_uniform(m, M):

	return random.random()*(M - m) + m


def rotate(origin, point, angle):
	"""
	Rotate a point counterclockwise by a given angle around a given origin.
	
	The angle should be given in radians.
	"""
	ox, oy = origin
	px, py = point

	qx = ox + (math.cos(1*angle) * (px - ox) - math.sin(1*angle) * (py - oy))
	qy = oy + (math.sin(1*angle) * (px - ox) + math.cos(1*angle) * (py - oy))
	return np.array([qx, qy])