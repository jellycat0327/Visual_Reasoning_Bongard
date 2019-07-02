'''

###############################################################################
Author: Tanner Bohn, Joey

Purpose: used to generate random Bongard-like image tiles for training a ML
          model which is then used to produce features for solving Bongard
          problems

Usage:
```
from TileHandler import *

# set shape value to None to prevent drawing that shape
TH = TileHandler(n_shapes = 1, size = 300, line_k_range = (1, 2),
                 circle = True, dot = True, curve = True, polygon_k_range = None,
                 ellipses = True, equilateral_polygon_range = (3, 5))

new_tile = TH.generate_tile()

# to view the tile
new_tile['img'].show()

# to convert the tile img and description to numpy arrays
img_vec, description_vec = TH.preprocess_tile(new_tile)


```

###############################################################################
'''
from __future__ import print_function


#import matplotlib.pyplot as plt
import math
import random
import numpy as np
from PIL import Image, ImageFilter, ImageDraw


class TileHandler:
    
    def __init__(self, n_shapes_range = (1, 1), size = 300, line_k_range = None,
                 circle = None, dot = None, curve = None, polygon_k_range = None,
                 ellipses = None, equilateral_polygon_range = None):
        # if shape parameter is None, then do not draw that shape
        self.n_shapes_range = n_shapes_range

        self.size = size

        self.line_k_range = line_k_range

        self.polygon_k_range = polygon_k_range

        self.ellipses = ellipses

        self.circle = circle

        self.dot = dot

        self.curve = curve

        self.equilateral_polygon_range = equilateral_polygon_range

        self.valid_shapes = []

        if line_k_range:
            self.valid_shapes.append("line")

        if circle:
            self.valid_shapes.append('circle')

        if curve:
            self.valid_shapes.append("curve")

        if dot:
            self.valid_shapes.append("dot")

        if polygon_k_range:
            self.valid_shapes.append("polygon")

        if ellipses:
            self.valid_shapes.append("ellipses")

        if equilateral_polygon_range:
            self.valid_shapes.append("equilateral_polygon")



    def vectorize_img(self, img):
        dim = img.size[0]
        pix = np.array(img)/255.
        img_vec = pix.reshape(dim, dim, 1)
        return np.array(img_vec, dtype=np.float16)

    def get_description_vec(self, tile):
        description_vec = np.array([tile['description'][key] for key in sorted(tile['description'])])
        return description_vec
             
    def preprocess_tile(self, tile):
        img_vec = self.vectorize_img(tile['img'])
        description_vec = self.get_description_vec(tile)
        return img_vec, description_vec
    #this function generates 4 lables [filled/not, big/small, up/down, left/right]
    def generate_tile_v3(self):

        # https://pillow.readthedocs.io/en/3.1.x/reference/ImageDraw.html

        # http://cglab.ca/~sander/misc/ConvexGeneration/convex.html

        # size: side length of images



        size = self.size

        line_k_range = self.line_k_range

        ellipses = self.ellipses

        polygon_k_range = self.polygon_k_range

        equilateral_polygon_range = self.equilateral_polygon_range

        dot = self.dot

        curve = self.curve

        circle = self.circle


        description = dict()
        #assume each tile only has one shape with default 4 classes
        description['filled_or_not'] = 0
        #description['unfilled'] = 0
        description['up_or_down'] = 0
        #description['down'] = 0
        description['left_or_right'] = 0
        #description['right'] = 0
        description['small_or_big'] = 0
        #description['big'] = 0
        all_x_coords = []
        all_y_coords = []
        up_threshold = size*0.5 #min y_coords <= up_threshold
        down_threshold = size*0.5 #max y_coords >= down_threshold
        left_threshold = size*0.5 #max x_coords <=  left_threshold
        right_threshold = size*0.5 #min x_coords >= right_threshold
        small_threshold = size/4 #max radius <= small_threshold(hard to calculate area)
        big_threshold = size/4 #min radius >= big_threshold

        img_size = (size, size)

        img = Image.new('L', (img_size), "white")

        img_draw = ImageDraw.Draw(img, 'L')

        for i_shape in range(random.randint(*self.n_shapes_range)):

            shape_type = random.choice(self.valid_shapes)

            if (shape_type == "line") :

                k = random.randint(line_k_range[0], line_k_range[1]) 
                
                k_p = k + 2
                
                radius = rand_uniform(0.1, 0.2)

                origin = (rand_uniform(radius, 1.-radius), rand_uniform(radius, 1.-radius))

                
                # Choose a random point on the circle and start rotating
                # (starting point chosen by rotating the top point by random angle)
                top_point = (origin[0], (origin[1] + radius))
                random_angle = rand_uniform(0, 2*math.pi)
                starting_point = rotate(origin, top_point, random_angle)
                
                rotation_angles = np.random.uniform(1, 6, k_p)
                rotation_angles = rotation_angles / np.sum(rotation_angles)
                rotation_angles = [sum(rotation_angles[:i]) for i in range(0, len(rotation_angles))]
                
                coords = [rotate(origin, starting_point, angle*2*math.pi) for angle in rotation_angles[:k+1]]


                # convert coords to pixels
                coords = [(int(x*size), int(y*size)) for x, y in coords]

                img_draw.line(coords, width=3)
                

                #description['line_'+str(k)] += 1

            elif (shape_type == "polygon"):
                
                # number of equilateral polygon sides
                k = random.randint(polygon_k_range[0], polygon_k_range[1])
                
                radius = rand_uniform(0.1, 0.4)
                if int(radius*size) >= big_threshold:
                    description['small_or_big'] = 1
                #this is the center of the shape
                origin = (rand_uniform(radius, 1.-radius), rand_uniform(radius, 1.-radius))
                if origin[0]*size >= right_threshold:
                    description['left_or_right'] = 1
                if origin[1]*size >= down_threshold:
                    description['up_or_down'] = 1
                # Choose a random point on the circle and start rotating
                # (starting point chosen by rotating the top point by random angle)
                top_point = (origin[0], (origin[1] + radius))
                random_angle = rand_uniform(0, 2*math.pi)
                starting_point = rotate(origin, top_point, random_angle)
                
                rotation_angles = np.random.uniform(1, 4, k)
                rotation_angles = rotation_angles / np.sum(rotation_angles)
                rotation_angles = [sum(rotation_angles[:i]) for i in range(0, len(rotation_angles))]
                
                coords = [rotate(origin, starting_point, angle*2*math.pi) for angle in rotation_angles]


                # convert coords to pixels
                coords = [(int(x*size), int(y*size)) for x, y in coords]
                filled = random.random() > 0.5
                if filled:
                    img_draw.polygon(coords, fill="black", outline="black")
                else:
                    coords.append(coords[0])
                    img_draw.line(coords, width=3)

                    description['filled_or_not'] = 1



            elif (shape_type == 'dot'):

                radius = rand_uniform(0.02, 0.04)
                center = (rand_uniform(radius, 1.-radius), rand_uniform(radius, 1.-radius))

                top_left = (center[0] - radius, center[1] - radius)
                bot_right = (center[0] + radius, center[1] + radius)
                
                coords = [top_left, bot_right]

                # convert coords to pixels
                coords = [(int(x*size), int(y*size)) for x, y in coords]

                img_draw.ellipse(coords, fill="black", outline="black")


                description['dot'] += 1


            elif (shape_type == 'curve'):
                
                radius_x = rand_uniform(0.075, 0.1)
                radius_y = rand_uniform(0.125, 0.2)
                
                if random.random() < 0.5:
                    radius_x, radius_y,  = radius_y, radius_x
                
                center = (rand_uniform(radius_x, 1.-radius_x), rand_uniform(radius_y, 1.-radius_y))

                top_left = (center[0] - radius_x, center[1] - radius_y)
                bot_right = (center[0] + radius_x, center[1] + radius_y) 

                coords = [top_left, bot_right]
                
                start_angle = rand_uniform(0,360)
                end_angle = start_angle + rand_uniform(45, 270)

                # convert coords to pixels
                coords = [(int(x*size), int(y*size)) for x, y in coords]
                img_draw.arc(coords, start_angle, end_angle, fill = None)

                description['curve'] += 1


            elif ( shape_type == "ellipses" ) :

                radius_x = rand_uniform(0.075, 0.2)
                radius_y = rand_uniform(0.125, 0.4)
                if int(max(radius_x,radius_y)*size) >= big_threshold:
                    description['small_or_big'] = 1
                    
                if random.random() < 0.5:
                    radius_x, radius_y,  = radius_y, radius_x
                
                center = (rand_uniform(radius_x, 1.-radius_x), rand_uniform(radius_y, 1.-radius_y))
                if center[0]*size >= right_threshold:
                    description['left_or_right'] = 1
                if center[1]*size >= down_threshold:
                    description['up_or_down'] = 1
                top_left = (center[0] - radius_x, center[1] - radius_y)
                bot_right = (center[0] + radius_x, center[1] + radius_y) 

                coords = [top_left, bot_right]

                # convert coords to pixels
                coords = [(int(x*size), int(y*size)) for x, y in coords]
                filled = random.random() > 0.5
                if filled:
                    img_draw.ellipse(coords, fill="black", outline="black")
                else:
                    img = draw_ellipse(img, coords, width=3, outline='black', antialias=4)

                    description['filled_or_not'] = 1

            elif shape_type == "circle":

                radius = rand_uniform(0.075, 0.4)
                #check area
                if int(radius*size)**2*math.pi >= size**2*0.5:
                    description['small_or_big'] = 1
                center = (rand_uniform(radius, 1.-radius), rand_uniform(radius, 1.-radius))
                if center[0]*size >= right_threshold:
                    description['left_or_right'] = 1
                if center[1]*size >= down_threshold:
                    description['up_or_down'] = 1
                top_left = (center[0] - radius, center[1] - radius)
                bot_right = (center[0] + radius, center[1] + radius)

                coords = [top_left, bot_right]

                # convert coords to pixels
                coords = [(int(x*size), int(y*size)) for x, y in coords]
                filled = random.random() > 0.5
                if filled:
                    img_draw.ellipse(coords, fill="black", outline="black")

                else:
                    img = draw_ellipse(img, coords, width = 2.5, outline='black', antialias=4)

                    description['filled_or_not'] = 1

            elif ( shape_type == 'equilateral_polygon' ):

                # number of equilateral polygon sides
                k = random.randint(equilateral_polygon_range[0], equilateral_polygon_range[1])
                
                radius = rand_uniform(0.1, 0.4)
                if int(radius*size) >= big_threshold:
                    description['small_or_big'] = 1
                    
                origin = (rand_uniform(radius, 1.-radius), rand_uniform(radius, 1.-radius))
                #print("origin point coords:", [(int(x*size), int(y*size)) for x, y in [origin]])
                if origin[0]*size >= right_threshold:
                    description['left_or_right'] = 1
                if origin[1]*size >= down_threshold:
                    description['up_or_down'] = 1
                # Choose a random point on the circle and start rotating
                # (starting point chosen by rotating the top point by random angle)
                top_point = (origin[0], (origin[1] + radius))
                random_angle = rand_uniform(0, 2*math.pi)
                starting_point = rotate(origin, top_point, random_angle)

                coords = [rotate(origin, starting_point, i_p*2*math.pi/k) for i_p in range(k)]
                
                # convert coords to pixels
                coords = [(int(x*size), int(y*size)) for x, y in coords]
                #print("final coords:", coords)
                filled = random.random() > 0.5
                if filled:
                    img_draw.polygon(coords, fill="black", outline="black")

                else:
                    coords.append(coords[0])
                    img_draw.line(coords, width=3)

                    description['filled_or_not'] = 1

        del img_draw

        return {"img": img, "description": description}

    def generate_tile_new(self):

        # https://pillow.readthedocs.io/en/3.1.x/reference/ImageDraw.html

        # http://cglab.ca/~sander/misc/ConvexGeneration/convex.html

        # size: side length of images



        size = self.size

        line_k_range = self.line_k_range

        ellipses = self.ellipses

        polygon_k_range = self.polygon_k_range

        equilateral_polygon_range = self.equilateral_polygon_range

        dot = self.dot

        curve = self.curve

        circle = self.circle


        description = dict()
        #assume each tile only has one shape with 8 classes
        description['filled'] = 0
        description['unfilled'] = 0
        description['up'] = 0
        description['down'] = 0
        description['left'] = 0
        description['right'] = 0
        description['small'] = 0
        description['big'] = 0
        all_x_coords = []
        all_y_coords = []
        up_threshold = size*0.4 #min y_coords <= up_threshold
        down_threshold = size*0.6 #max y_coords >= down_threshold
        left_threshold = size*0.4 #max x_coords <=  left_threshold
        right_threshold = size*0.6 #min x_coords >= right_threshold
        small_threshold = size/4 #max radius <= small_threshold(hard to calculate area)
        big_threshold = size/4 #min radius >= big_threshold

        img_size = (size, size)

        img = Image.new('L', (img_size), "white")

        img_draw = ImageDraw.Draw(img, 'L')

        for i_shape in range(random.randint(*self.n_shapes_range)):

            shape_type = random.choice(self.valid_shapes)

            if (shape_type == "line") :

                k = random.randint(line_k_range[0], line_k_range[1]) 
                
                k_p = k + 2
                
                radius = rand_uniform(0.1, 0.2)

                origin = (rand_uniform(radius, 1.-radius), rand_uniform(radius, 1.-radius))

                
                # Choose a random point on the circle and start rotating
                # (starting point chosen by rotating the top point by random angle)
                top_point = (origin[0], (origin[1] + radius))
                random_angle = rand_uniform(0, 2*math.pi)
                starting_point = rotate(origin, top_point, random_angle)
                
                rotation_angles = np.random.uniform(1, 6, k_p)
                rotation_angles = rotation_angles / np.sum(rotation_angles)
                rotation_angles = [sum(rotation_angles[:i]) for i in range(0, len(rotation_angles))]
                
                coords = [rotate(origin, starting_point, angle*2*math.pi) for angle in rotation_angles[:k+1]]


                # convert coords to pixels
                coords = [(int(x*size), int(y*size)) for x, y in coords]

                img_draw.line(coords, width=3)
                

                #description['line_'+str(k)] += 1

            elif (shape_type == "polygon"):
                
                # number of equilateral polygon sides
                k = random.randint(polygon_k_range[0], polygon_k_range[1])
                
                radius = rand_uniform(0.1, 0.4)
                if int(radius*size) <= small_threshold:
                    description['small'] = 1
                elif int(radius*size) >= big_threshold:
                    description['big'] = 1
                #this is the center of the shape
                origin = (rand_uniform(radius, 1.-radius), rand_uniform(radius, 1.-radius))
                if origin[0]*size <= left_threshold:
                    description['left'] = 1
                elif origin[0]*size > right_threshold:
                    description['right'] = 1
                if origin[1]*size <= up_threshold:
                    description['up'] = 1
                elif origin[1]*size > down_threshold:
                    description['down'] = 1
                # Choose a random point on the circle and start rotating
                # (starting point chosen by rotating the top point by random angle)
                top_point = (origin[0], (origin[1] + radius))
                random_angle = rand_uniform(0, 2*math.pi)
                starting_point = rotate(origin, top_point, random_angle)
                
                rotation_angles = np.random.uniform(1, 4, k)
                rotation_angles = rotation_angles / np.sum(rotation_angles)
                rotation_angles = [sum(rotation_angles[:i]) for i in range(0, len(rotation_angles))]
                
                coords = [rotate(origin, starting_point, angle*2*math.pi) for angle in rotation_angles]


                # convert coords to pixels
                coords = [(int(x*size), int(y*size)) for x, y in coords]
                #check coords to set label
#                 for x, y in coords:
#                     all_x_coords.append(x)
#                     all_y_coords.append(y)
#                 if max(all_y_coords) <= up_threshold:
#                     description['up'] = 1
#                 elif min(all_y_coords) >= down_threshold:
#                     description['down'] = 1
#                 if max(all_x_coords) <= left_threshold:
#                     description['left'] = 1
#                 elif min(all_x_coords) >= right_threshold:
#                     description['right'] = 1
                filled = random.random() > 0.5
                if filled:
                    img_draw.polygon(coords, fill="black", outline="black")

                    description['filled'] = 1
                else:
                    coords.append(coords[0])
                    img_draw.line(coords, width=3)

                    description['unfilled'] = 1



            elif (shape_type == 'dot'):

                radius = rand_uniform(0.02, 0.04)
                center = (rand_uniform(radius, 1.-radius), rand_uniform(radius, 1.-radius))

                top_left = (center[0] - radius, center[1] - radius)
                bot_right = (center[0] + radius, center[1] + radius)
                
                coords = [top_left, bot_right]

                # convert coords to pixels
                coords = [(int(x*size), int(y*size)) for x, y in coords]

                img_draw.ellipse(coords, fill="black", outline="black")


                description['dot'] += 1


            elif (shape_type == 'curve'):
                
                radius_x = rand_uniform(0.075, 0.1)
                radius_y = rand_uniform(0.125, 0.2)
                
                if random.random() < 0.5:
                    radius_x, radius_y,  = radius_y, radius_x
                
                center = (rand_uniform(radius_x, 1.-radius_x), rand_uniform(radius_y, 1.-radius_y))

                top_left = (center[0] - radius_x, center[1] - radius_y)
                bot_right = (center[0] + radius_x, center[1] + radius_y) 

                coords = [top_left, bot_right]
                
                start_angle = rand_uniform(0,360)
                end_angle = start_angle + rand_uniform(45, 270)

                # convert coords to pixels
                coords = [(int(x*size), int(y*size)) for x, y in coords]
                #check coords to set label
#                 for x, y in coords:
#                     all_x_coords.append(x)
#                     all_y_coords.append(y)
#                 if max(all_y_coords) <= up_threshold:
#                     description['up'] = 1
#                 elif min(all_y_coords) >= down_threshold:
#                     description['down'] = 1
#                 if max(all_x_coords) <= left_threshold:
#                     description['left'] = 1
#                 elif min(all_x_coords) >= right_threshold:
#                     description['right'] = 1
                img_draw.arc(coords, start_angle, end_angle, fill = None)

                description['curve'] += 1


            elif ( shape_type == "ellipses" ) :

                radius_x = rand_uniform(0.075, 0.2)
                radius_y = rand_uniform(0.125, 0.4)
                if int(min(radius_x,radius_y)*size) <= small_threshold:
                    description['small'] = 1
                elif int(max(radius_x,radius_y)*size) >= big_threshold:
                    description['big'] = 1
                    
                if random.random() < 0.5:
                    radius_x, radius_y,  = radius_y, radius_x
                
                center = (rand_uniform(radius_x, 1.-radius_x), rand_uniform(radius_y, 1.-radius_y))
                if center[0]*size <= left_threshold:
                    description['left'] = 1
                elif center[0]*size > right_threshold:
                    description['right'] = 1
                if center[1]*size <= up_threshold:
                    description['up'] = 1
                elif center[1]*size > down_threshold:
                    description['down'] = 1
                top_left = (center[0] - radius_x, center[1] - radius_y)
                bot_right = (center[0] + radius_x, center[1] + radius_y) 

                coords = [top_left, bot_right]

                # convert coords to pixels
                coords = [(int(x*size), int(y*size)) for x, y in coords]
                #check coords to set label
#                 for x, y in coords:
#                     all_x_coords.append(x)
#                     all_y_coords.append(y)
#                 if max(all_y_coords) <= up_threshold:
#                     description['up'] = 1
#                 elif min(all_y_coords) >= down_threshold:
#                     description['down'] = 1
#                 if max(all_x_coords) <= left_threshold:
#                     description['left'] = 1
#                 elif min(all_x_coords) >= right_threshold:
#                     description['right'] = 1
                filled = random.random() > 0.5
                if filled:
                    img_draw.ellipse(coords, fill="black", outline="black")

                    description['filled'] = 1

                else:
                    img = draw_ellipse(img, coords, width=3, outline='black', antialias=4)

                    description['unfilled'] = 1

            elif shape_type == "circle":

                radius = rand_uniform(0.075, 0.4)
                #check area
                if int(radius*size)**2*math.pi <= size**2*0.5:
                    description['small'] = 1
                elif int(radius*size)**2*math.pi >= size**2*0.5:
                    description['big'] = 1
                center = (rand_uniform(radius, 1.-radius), rand_uniform(radius, 1.-radius))
                if center[0]*size <= left_threshold:
                    description['left'] = 1
                elif center[0]*size > right_threshold:
                    description['right'] = 1
                if center[1]*size <= up_threshold:
                    description['up'] = 1
                elif center[1]*size > down_threshold:
                    description['down'] = 1
                top_left = (center[0] - radius, center[1] - radius)
                bot_right = (center[0] + radius, center[1] + radius)

                coords = [top_left, bot_right]

                # convert coords to pixels
                coords = [(int(x*size), int(y*size)) for x, y in coords]
                #check coords to set label
#                 for x, y in coords:
#                     all_x_coords.append(x)
#                     all_y_coords.append(y)
#                 if max(all_y_coords) <= up_threshold:
#                     description['up'] = 1
#                 elif min(all_y_coords) >= down_threshold:
#                     description['down'] = 1
#                 if max(all_x_coords) <= left_threshold:
#                     description['left'] = 1
#                 elif min(all_x_coords) >= right_threshold:
#                     description['right'] = 1
                filled = random.random() > 0.5
                if filled:
                    img_draw.ellipse(coords, fill="black", outline="black")

                    description['filled'] = 1

                else:
                    img = draw_ellipse(img, coords, width = 2.5, outline='black', antialias=4)

                    description['unfilled'] = 1


            elif ( shape_type == 'equilateral_polygon' ):

                # number of equilateral polygon sides
                k = random.randint(equilateral_polygon_range[0], equilateral_polygon_range[1])
                
                radius = rand_uniform(0.1, 0.4)
                if int(radius*size) <= small_threshold:
                    description['small'] = 1
                elif int(radius*size) >= big_threshold:
                    description['big'] = 1
                    
                origin = (rand_uniform(radius, 1.-radius), rand_uniform(radius, 1.-radius))
                #print("origin point coords:", [(int(x*size), int(y*size)) for x, y in [origin]])
                if origin[0]*size <= left_threshold:
                    description['left'] = 1
                elif origin[0]*size > right_threshold:
                    description['right'] = 1
                if origin[1]*size <= up_threshold:
                    description['up'] = 1
                elif origin[1]*size > down_threshold:
                    description['down'] = 1
                # Choose a random point on the circle and start rotating
                # (starting point chosen by rotating the top point by random angle)
                top_point = (origin[0], (origin[1] + radius))
                random_angle = rand_uniform(0, 2*math.pi)
                starting_point = rotate(origin, top_point, random_angle)

                coords = [rotate(origin, starting_point, i_p*2*math.pi/k) for i_p in range(k)]
                
                # convert coords to pixels
                coords = [(int(x*size), int(y*size)) for x, y in coords]
                #check coords to set label
#                 for x, y in coords:
#                     all_x_coords.append(x)
#                     all_y_coords.append(y)
#                 if max(all_y_coords) <= up_threshold:
#                     description['up'] = 1
#                 elif min(all_y_coords) >= down_threshold:
#                     description['down'] = 1
#                 if max(all_x_coords) <= left_threshold:
#                     description['left'] = 1
#                 elif min(all_x_coords) >= right_threshold:
#                     description['right'] = 1
                #print("final coords:", coords)
                filled = random.random() > 0.5
                if filled:
                    img_draw.polygon(coords, fill="black", outline="black")

                    description['filled'] = 1
                else:
                    coords.append(coords[0])
                    img_draw.line(coords, width=3)

                    description['unfilled'] = 1

        del img_draw

        return {"img": img, "description": description}
    def generate_tile(self, filled=True):

        # https://pillow.readthedocs.io/en/3.1.x/reference/ImageDraw.html

        # http://cglab.ca/~sander/misc/ConvexGeneration/convex.html

        # size: side length of images



        size = self.size

        line_k_range = self.line_k_range

        ellipses = self.ellipses

        polygon_k_range = self.polygon_k_range

        equilateral_polygon_range = self.equilateral_polygon_range

        dot = self.dot

        curve = self.curve

        circle = self.circle


        description = dict()

        if self.line_k_range:
            for k in range(line_k_range[0], line_k_range[1]+1):
                description['line_{}'.format(k)] = 0


        if self.polygon_k_range:
            for k in range(polygon_k_range[0], polygon_k_range[1]+1):
                description['polygon_filled_{}'.format(k)] = 0

            for k in range(polygon_k_range[0], polygon_k_range[1]+1):
                description['polygon_unfilled_{}'.format(k)] = 0
        

        if self.dot:
            description['dot'] = 0

        if self.curve:
            description['curve'] = 0

        if self.circle:
            description['circle_filled'] = 0

            description['circle_unfilled'] = 0

        if self.ellipses:
            description['ellipses_filled'] = 0

            description['ellipses_unfilled'] = 0

        if self.equilateral_polygon_range:
            for k in range(equilateral_polygon_range[0], equilateral_polygon_range[1]+1):

                description['equilateral_polygon_filled_{}'.format(k)] = 0

            for k in range(equilateral_polygon_range[0], equilateral_polygon_range[1]+1):

                description['equilateral_polygon_unfilled_{}'.format(k)] = 0

        img_size = (size, size)

        img = Image.new('L', (img_size), "white")

        img_draw = ImageDraw.Draw(img, 'L')

        for i_shape in range(random.randint(*self.n_shapes_range)):

            shape_type = random.choice(self.valid_shapes)

            if (shape_type == "line") :

                k = random.randint(line_k_range[0], line_k_range[1]) 
                
                k_p = k + 2
                
                radius = rand_uniform(0.1, 0.2)

                origin = (rand_uniform(radius, 1.-radius), rand_uniform(radius, 1.-radius))

                
                # Choose a random point on the circle and start rotating
                # (starting point chosen by rotating the top point by random angle)
                top_point = (origin[0], (origin[1] + radius))
                random_angle = rand_uniform(0, 2*math.pi)
                starting_point = rotate(origin, top_point, random_angle)
                
                rotation_angles = np.random.uniform(1, 6, k_p)
                rotation_angles = rotation_angles / np.sum(rotation_angles)
                rotation_angles = [sum(rotation_angles[:i]) for i in range(0, len(rotation_angles))]
                
                coords = [rotate(origin, starting_point, angle*2*math.pi) for angle in rotation_angles[:k+1]]


                # convert coords to pixels
                coords = [(int(x*size), int(y*size)) for x, y in coords]

                img_draw.line(coords, width=3)
                

                #description['line_'+str(k)] += 1

            elif (shape_type == "polygon"):
                
                # number of equilateral polygon sides
                k = random.randint(polygon_k_range[0], polygon_k_range[1])
                
                radius = rand_uniform(0.1, 0.2)

                origin = (rand_uniform(radius, 1.-radius), rand_uniform(radius, 1.-radius))

                
                # Choose a random point on the circle and start rotating
                # (starting point chosen by rotating the top point by random angle)
                top_point = (origin[0], (origin[1] + radius))
                random_angle = rand_uniform(0, 2*math.pi)
                starting_point = rotate(origin, top_point, random_angle)
                
                rotation_angles = np.random.uniform(1, 4, k)
                rotation_angles = rotation_angles / np.sum(rotation_angles)
                rotation_angles = [sum(rotation_angles[:i]) for i in range(0, len(rotation_angles))]
                
                coords = [rotate(origin, starting_point, angle*2*math.pi) for angle in rotation_angles]


                # convert coords to pixels
                coords = [(int(x*size), int(y*size)) for x, y in coords]
                #filled = random.random() > 0.5
                if filled:
                    img_draw.polygon(coords, fill="black", outline="black")

                    description['polygon_filled_'+str(k)] += 1
                else:
                    coords.append(coords[0])
                    img_draw.line(coords, width=3)

                    description['polygon_unfilled_'+str(k)] += 1



            elif (shape_type == 'dot'):

                radius = rand_uniform(0.02, 0.04)
                center = (rand_uniform(radius, 1.-radius), rand_uniform(radius, 1.-radius))

                top_left = (center[0] - radius, center[1] - radius)
                bot_right = (center[0] + radius, center[1] + radius)
                
                coords = [top_left, bot_right]

                # convert coords to pixels
                coords = [(int(x*size), int(y*size)) for x, y in coords]

                img_draw.ellipse(coords, fill="black", outline="black")


                description['dot'] += 1


            elif (shape_type == 'curve'):
                
                radius_x = rand_uniform(0.075, 0.1)
                radius_y = rand_uniform(0.125, 0.2)
                
                if random.random() < 0.5:
                    radius_x, radius_y,  = radius_y, radius_x
                
                center = (rand_uniform(radius_x, 1.-radius_x), rand_uniform(radius_y, 1.-radius_y))

                top_left = (center[0] - radius_x, center[1] - radius_y)
                bot_right = (center[0] + radius_x, center[1] + radius_y) 

                coords = [top_left, bot_right]
                
                start_angle = rand_uniform(0,360)
                end_angle = start_angle + rand_uniform(45, 270)

                # convert coords to pixels
                coords = [(int(x*size), int(y*size)) for x, y in coords]

                img_draw.arc(coords, start_angle, end_angle, fill = None)

                description['curve'] += 1


            elif ( shape_type == "ellipses" ) :

                radius_x = rand_uniform(0.075, 0.1)
                radius_y = rand_uniform(0.125, 0.2)
                
                if random.random() < 0.5:
                    radius_x, radius_y,  = radius_y, radius_x
                
                center = (rand_uniform(radius_x, 1.-radius_x), rand_uniform(radius_y, 1.-radius_y))

                top_left = (center[0] - radius_x, center[1] - radius_y)
                bot_right = (center[0] + radius_x, center[1] + radius_y) 

                coords = [top_left, bot_right]

                # convert coords to pixels
                coords = [(int(x*size), int(y*size)) for x, y in coords]

                #filled = random.random() > 0.5
                if filled:
                    img_draw.ellipse(coords, fill="black", outline="black")

                    description['ellipses_filled'] += 1

                else:
                    img = draw_ellipse(img, coords, width=3, outline='black', antialias=4)

                    description['ellipses_unfilled'] += 1

            elif shape_type == "circle":

                radius = rand_uniform(0.075, 0.2)
                center = (rand_uniform(radius, 1.-radius), rand_uniform(radius, 1.-radius))

                top_left = (center[0] - radius, center[1] - radius)
                bot_right = (center[0] + radius, center[1] + radius)

                coords = [top_left, bot_right]

                # convert coords to pixels
                coords = [(int(x*size), int(y*size)) for x, y in coords]

                #filled = random.random() > 0.5
                if filled:
                    img_draw.ellipse(coords, fill="black", outline="black")

                    description['circle_filled'] += 1

                else:
                    img = draw_ellipse(img, coords, width = 2.5, outline='black', antialias=4)

                    description['circle_unfilled'] += 1


            elif ( shape_type == 'equilateral_polygon' ):

                # number of equilateral polygon sides
                k = random.randint(equilateral_polygon_range[0], equilateral_polygon_range[1])
                
                radius = rand_uniform(0.1, 0.2)

                origin = (rand_uniform(radius, 1.-radius), rand_uniform(radius, 1.-radius))

                
                # Choose a random point on the circle and start rotating
                # (starting point chosen by rotating the top point by random angle)
                top_point = (origin[0], (origin[1] + radius))
                random_angle = rand_uniform(0, 2*math.pi)
                starting_point = rotate(origin, top_point, random_angle)

                coords = [rotate(origin, starting_point, i_p*2*math.pi/k) for i_p in range(k)]

                # convert coords to pixels
                coords = [(int(x*size), int(y*size)) for x, y in coords]

                #filled = random.random() > 0.5
                if filled:
                    img_draw.polygon(coords, fill="black", outline="black")

                    description['equilateral_polygon_filled_'+str(k)] += 1
                else:
                    coords.append(coords[0])
                    img_draw.line(coords, width=3)

                    description['equilateral_polygon_unfilled_'+str(k)] += 1

        del img_draw

        return {"img": img, "description": description}

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