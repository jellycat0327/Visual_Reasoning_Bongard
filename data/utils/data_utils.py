import numpy as np
from PIL import Image


def vectorize_img(img):
    dim = img.size[0]
    pix = np.array(img)/255.
    img_vec = pix.reshape(dim, dim, 1)
    return np.array(img_vec, dtype=np.float16)

def flatten_img(img):
    pix = np.array(img)/255.
    return np.array(pix.flatten(), dtype=np.float16)


def img_to_fft(img, flatten=False):

    dim = img.size[0]

    img_vec = 1. - np.array(img)/255.

    ftimage = np.fft.fft2(img_vec, norm="ortho")

    ftimage = abs(np.fft.fftshift(ftimage))


    if flatten:
        # return flat vector
        return np.array(ftimage.flatten(), dtype=np.float16)
    else:
        # return conv input form
        return np.array(ftimage.reshape(dim, dim, 1), dtype=np.float16)

def get_real_BP_data(BDA, num, resizing=None, use_fft=False, flatten=False):

    X_train = []
    y_train = []

    X_test = []
    y_test = []

    for field, label, i_t_range in [('real_left', 0, 6), ('real_right', 1, 6), ('manual_left', 0, 1), ('manual_right', 1, 1)]:
        for i_t in range(i_t_range):
            img = BDA.bongard_problems[num][field][i_t]
            if resizing != None:
                img = img.resize((resizing, resizing), Image.LANCZOS)

            x = None
            if use_fft:
                x = img_to_fft(img, flatten=flatten)
            else:
                if not flatten:
                    x = 1. - vectorize_img(img)
                else:
                    x = 1. - flatten_img(img) 

            if 'real' in field:
                X_train.append(x)
                y_train.append(label)	
            else: # manual
                X_test.append(x)
                y_test.append(label)


    return X_train, y_train, X_test, y_test

def get_data(BPG, bp_num, nb_train, nb_val, nb_test, resizing=None, flatten=False, use_fft=False):

    #resizing = BPG.img_size[0]

    X_train, X_val, X_test = [], [], []
    y_train, y_val, y_test = [], [], []

    for partition, nb in [("train", nb_train), ("val", nb_val), ("test", nb_test)]:

        for i in range(nb):
            side = "left" if i%2==0 else "right"	

            #x = BPG.flatten_img(BPG.draw_tile(61, side).resize((resizing, resizing), Image.LANCZOS))
            img = BPG.draw_tile(bp_num, side)
            if resizing != None:
                img = img.resize((resizing, resizing), Image.LANCZOS)

            x = None
            if use_fft:
                x = img_to_fft(img, flatten=flatten)
            else:
                if not flatten:
                    x = BPG.vectorize_img(img)
                else:
                    x = BPG.flatten_img(img) 

            label = 0 if side == "left" else 1

            if partition == "train":
                X_train.append(x)
                y_train.append(label)
            elif partition == "val":
                X_val.append(x)
                y_val.append(label)
            elif partition == "test":
                X_test.append(x)
                y_test.append(label)


    return np.array(X_train), np.array(y_train), np.array(X_val), np.array(y_val), np.array(X_test), np.array(y_test)

def get_data_from_TH(TH, nb_train, nb_val, nb_test, resizing=None, flatten=False, use_fft=False):

    #resizing = BPG.img_size[0]

    X_train, X_val, X_test = [], [], []
    y_train, y_val, y_test = [], [], []

    for partition, nb in [("train", nb_train), ("val", nb_val), ("test", nb_test)]:

        for i in range(nb):

            img_data = TH.generate_tile_new()
            img = img_data['img']

            #description: {'filled': 0, 'right': 0, 'unfilled': 1, 'big': 1, 'down': 0, 'small': 0, 'left': 0, 'up': 0}
            img_description = img_data['description']
            if resizing != None:
                img = img.resize((resizing, resizing), Image.LANCZOS)

            x = None
            if use_fft:
                x = img_to_fft(img, flatten=flatten)
            else:
                if not flatten:
                    x = 1. - TH.vectorize_img(img)
                else:
                    x = 1. - TH.flatten_img(img) 

            label = np.array([img_description[k] for k in sorted(img_description)])

            if partition == "train":
                X_train.append(x)
                y_train.append(label)
            elif partition == "val":
                X_val.append(x)
                y_val.append(label)
            elif partition == "test":
                X_test.append(x)
                y_test.append(label)


    return np.array(X_train), np.array(y_train), np.array(X_val), np.array(y_val), np.array(X_test), np.array(y_test)