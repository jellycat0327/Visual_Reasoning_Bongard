
import sys
import os

import numpy as np

from utils.TileHandler import *
from utils.BPGenerator import *

from utils import data_utils

from sklearn.metrics import roc_auc_score
import itertools

class Evaluator:


    def __init__(self, img_size=(48, 48), nb_unlabelled=1000, nb_labelled=100, nb_test=10000):

        self.img_size = img_size

        self.nb_unlabelled = nb_unlabelled
        self.nb_labelled = nb_labelled
        self.nb_test = nb_test

        self.BPG = BPGenerator(self.img_size)


        self.tile_handler = TileHandler(n_shapes_range = (1,1), size = self.img_size[0], polygon_k_range = None, equilateral_polygon_range = (3,6), circle=True, ellipses=True)

        self.basic_task_names = sorted(self.tile_handler.generate_tile_new()['description'].keys()) # ['fill', 'height', 'round', 'side', 'size']
        self.bongard_task_names = ["BP_1", "BP_2", "BP_3", "BP_5", "BP_6", "BP_7", "BP_8", "BP_13", "BP_22", "BP_61"]  #["BP_2", "BP_5", "BP_6", "BP_13", "BP_22"]  #
        # BP3 = fill/unfill
        # bp8 = left/right
        # bp2 = big/small
        # bp5 = round/polygon
        return


    def evaluate(self, model, verbose=0):


        task_scores = []

        for task in self.bongard_task_names:

            X_train_unlabelled, X_train_labelled, y_train_labelled = self.get_data(task, self.nb_unlabelled, self.nb_labelled)

            model.fit(X_train_unlabelled, X_train_labelled, y_train_labelled, verbose=verbose>1)

            _, X_test, y_test = self.get_data(task, nb_unlabelled=0, nb_labelled=self.nb_test)

            y_pred = model.predict(X_test, verbose=verbose>1)
            test_score = roc_auc_score(y_test, y_pred)

            task_scores.append((task, test_score))

            if verbose >= 1:
                print("TASK SCORE:")
                print("{}\t{:.4f}".format(task, test_score))


        return task_scores, np.average([v[1] for v in task_scores])

    def get_data(self, task, nb_unlabelled, nb_labelled):

        if task in self.basic_task_names:

            task_index = self.basic_task_names.index(task)

            good_split = False
            nb_tries = 0
            while not good_split:
                nb_tries += 1
                X_train, _, _, _, X_test, y_test = data_utils.get_data_from_TH(self.tile_handler, nb_train=nb_unlabelled, nb_val=0, nb_test=nb_labelled)
                y_test = y_test[:,task_index]

                nb_unique = len(set(y_test))
                if nb_unique == 2:
                    good_split = True
                    return X_train, X_test, y_test

                assert nb_tries < 10, "ERROR: too many attempts required for good data split"

        elif task in self.bongard_task_names:

            bp_num = task.split("_")[1]

            good_split = False
            nb_tries = 0
            while not good_split:
                nb_tries += 1
                X_train, _, _, _, X_test, y_test = data_utils.get_data(self.BPG, bp_num, nb_train=nb_unlabelled, nb_val=0, nb_test=nb_labelled)

                nb_unique = len(set(y_test))
                if nb_unique == 2:
                    good_split = True
                    return X_train, X_test, y_test

                assert nb_tries < 10, "ERROR: too many attempts required for good data split"