import numpy as np
import random
import csv
from scipy.stats import norm

from math import log
from operator import itemgetter


folds = 10
split = 0.2

confusion_matrix = np.zeros((2, 2), dtype=int)


class my_naive_bayes:

    def __init__(self, training_data):

        self.prior_dist_log = {}
        self.conditional_dist = {}
        self.classes = []

        if training_data is not None:
            self.train(training_data)

        return

    def train(self, data):

        num_training = len(data)

        data[data[:, -1].argsort()]
        classes = np.array(list(set([item[-1] for item in data])))
        self.classes = classes
        grouped_data = np.array(
            [[item[:-1] for item in data if item[-1] == class_name] for class_name in classes])

        for i, group in enumerate((grouped_data)):

            # generate distribution

            self.prior_dist_log[classes[i]] = log(len(group) / num_training)
            self.conditional_dist[classes[i]] = {}

            for j, column in enumerate(np.array(group).T):

                if j in [3, 4, 6, 8]:
                    column = [non_zero for non_zero in column if non_zero != 0]
                miu = np.mean(column)
                sigma = np.std(column)

                self.conditional_dist[classes[i]][j] = norm(miu, sigma)

        return

    def inference(self, data):

        # it works but I should never write code like this
        return max([[class_name, sum([log(max(np.finfo(float).tiny, distribution.pdf(data[i]))) for i, distribution in self.conditional_dist[class_name].items() if data[i] != 0 or i not in [3, 4, 6, 8]]
                                     ) + self.prior_dist_log[class_name]] for class_name in self.classes], key=itemgetter(1))[0]


def read_csv(path: str = "./data/datasets_14370_19291_pima-indians-diabetes.csv"):

    return np.genfromtxt(path, delimiter=',', dtype=np.uint8)


def main():

    all_training_data = read_csv()
    data_len = len(all_training_data)

    for i in range(folds):
        random.Random(i+12).shuffle(all_training_data)

        training = all_training_data[int(data_len * split):]
        valid = all_training_data[:int(data_len * split)]

        my_classifier = my_naive_bayes(training)

        for j, x in enumerate(valid):
            confusion_matrix[x[-1]][my_classifier.inference(x[:-1])] += 1

    with open("./performenceB.txt", "w") as f:
        f.write(str(confusion_matrix))
        f.write("\nerror: %" + str(100 *
                                   (confusion_matrix[0][1] + confusion_matrix[1][0]) / np.sum(confusion_matrix)))

    return


if __name__ == "__main__":

    main()
