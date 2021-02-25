import itertools
import numpy as np

class all_test_data():
    def __init__(self, Fouriers, sound_names):
        self.Fouriers = Fouriers
        self.sound_names = sound_names

    def for_kmeans(self, sound1, sound2):

        train_num = [0, 0]
        iterate_num = 0

        for i in range(len(self.sound_names)):
            if sound1 == self.sound_names[i]:
                train1 = self.Fouriers[40 * i + 35:40 * i + 40]
            if sound2 == self.sound_names[i]:
                train2 = self.Fouriers[40 * i + 35:40 * i + 40]

        try:
            if 'train1' not in locals():
                raise NameError('sound1 was not found.')
            if 'train2' not in locals():
                raise NameError('sound2 was not found.')
        except NameError as message:
            print(message)

        mixture = list()
        while True:
            first = train1[train_num[0]]
            second = train2[train_num[1]]

            try:
                if first.shape[-1] < 100:
                    raise TypeError("The data is too short.")
            except TypeError as message:
                print(message)

            if first.shape[-1] > 100:
                if first.shape[-1] < (iterate_num + 1) * 100:
                    first = np.concatenate(
                        (first[:, iterate_num * 100:], first[:, :(100 - (first.shape[-1] - iterate_num * 100))]),
                        axis=1)
                    second = np.concatenate(
                        (second[:, iterate_num * 100:], second[:, :(100 - (second.shape[-1] - iterate_num * 100))]),
                        axis=1)

                    train_num[1] += 1
                    if len(train2) == train_num[1]:
                        train_num[0] += 1
                        train_num[1] = 0
                        if len(train1) == train_num[0]:
                            train_num[0] = 0
                    iterate_num = 0
                else:
                    first = first[:, iterate_num * 100:(iterate_num + 1) * 100]
                    second = second[:, iterate_num * 100:(iterate_num + 1) * 100]
                    iterate_num += 1
            else:
                train_num[1] += 1
                if len(train2) == train_num[1]:
                    train_num[0] += 1
                    train_num[1] = 0
                    if len(train1) == train_num[0]:
                        train_num[0] = 0

            mix = first + second

            mixture.append(mix)

            if train_num == [0, 0] and iterate_num == 0:
                x_test = np.asarray(mixture)

                mixture = list()

                return x_test

    def for_predict(self, sound1, sound2):

        train_num = [0, 0]
        iterate_num = 0

        for i in range(len(self.sound_names)):
            if sound1 == self.sound_names[i]:
                train1 = self.Fouriers[40 * i + 35:40 * i + 40]
            if sound2 == self.sound_names[i]:
                train2 = self.Fouriers[40 * i + 35:40 * i + 40]

        try:
            if 'train1' not in locals():
                raise NameError('sound1 was not found.')
            if 'train2' not in locals():
                raise NameError('sound2 was not found.')
        except NameError as message:
            print(message)

        before = list()
        mixture = list()
        while True:
            first = train1[train_num[0]]
            second = train2[train_num[1]]

            try:
                if first.shape[-1] < 100:
                    raise TypeError("The data is too short.")
            except TypeError as message:
                print(message)

            if first.shape[-1] > 100:
                if first.shape[-1] < (iterate_num + 1) * 100:
                    first = np.concatenate(
                        (first[:, iterate_num * 100:], first[:, :(100 - (first.shape[-1] - iterate_num * 100))]),
                        axis=1)
                    second = np.concatenate(
                        (second[:, iterate_num * 100:], second[:, :(100 - (second.shape[-1] - iterate_num * 100))]),
                        axis=1)

                    train_num[1] += 1
                    if len(train2) == train_num[1]:
                        train_num[0] += 1
                        train_num[1] = 0
                        if len(train1) == train_num[0]:
                            train_num[0] = 0
                    iterate_num = 0
                else:
                    first = first[:, iterate_num * 100:(iterate_num + 1) * 100]
                    second = second[:, iterate_num * 100:(iterate_num + 1) * 100]
                    iterate_num += 1
            else:
                train_num[1] += 1
                if len(train2) == train_num[1]:
                    train_num[0] += 1
                    train_num[1] = 0
                    if len(train1) == train_num[0]:
                        train_num[0] = 0

            mix = first + second

            before.append([first, second])
            mixture.append(mix)

            if train_num == [0, 0] and iterate_num == 0:
                x_test1 = np.asarray(mixture)
                x_test2 = np.zeros(shape=(x_test1.shape[0], 2, 129, 100))
                y_test = np.asarray(before)

                mixture = list()
                before = list()

                return [x_test1, x_test2], y_test

    def generate_data(self, sound1, sound2):
        train_num = [0, 0]
        iterate_num = 0

        for i in range(len(self.sound_names)):
            if sound1 == self.sound_names[i]:
                train1 = self.Fouriers[40 * i + 35:40 * i + 40]
            if sound2 == self.sound_names[i]:
                train2 = self.Fouriers[40 * i + 35:40 * i + 40]

        try:
            if 'train1' not in locals():
                raise NameError('sound1 was not found.')
            if 'train2' not in locals():
                raise NameError('sound2 was not found.')
        except NameError as message:
            print(message)

        before = list()
        mixture = list()
        while True:
            first = train1[train_num[0]]
            second = train2[train_num[1]]

            try:
                if first.shape[-1] < 100:
                    raise TypeError("The data is too short.")
            except TypeError as message:
                print(message)

            if first.shape[-1] > 100:
                if first.shape[-1] < (iterate_num + 1) * 100:
                    first = np.concatenate(
                        (first[:, iterate_num * 100:], first[:, :(100 - (first.shape[-1] - iterate_num * 100))]),
                        axis=1)
                    second = np.concatenate(
                        (second[:, iterate_num * 100:], second[:, :(100 - (second.shape[-1] - iterate_num * 100))]),
                        axis=1)

                    train_num[1] += 1
                    if len(train2) == train_num[1]:
                        train_num[0] += 1
                        train_num[1] = 0
                        if len(train1) == train_num[0]:
                            train_num[0] = 0
                    iterate_num = 0
                else:
                    first = first[:, iterate_num * 100:(iterate_num + 1) * 100]
                    second = second[:, iterate_num * 100:(iterate_num + 1) * 100]
                    iterate_num += 1
            else:
                train_num[1] += 1
                if len(train2) == train_num[1]:
                    train_num[0] += 1
                    train_num[1] = 0
                    if len(train1) == train_num[0]:
                        train_num[0] = 0

            mix = first + second

            before.append([first, second])
            mixture.append(mix)

            if train_num == [0, 0] and iterate_num == 0:
                x_test = np.asarray(mixture)
                y_test = np.asarray(before)

                mixture = list()
                before = list()

                return x_test, y_test


class generator():
    def __init__(self, Fouriers, sound_names):
        self.train_num = [0, 0]
        self.iterate_num = 0
        self.Fouriers = Fouriers
        self.sound_names = sound_names
        return

    def generator_train(self, sound1, sound2, batch_size, initialize=True):
        if initialize:
            self.train_num = [0, 0]
            self.iterate_num = 0

        for i in range(len(self.sound_names)):
            if sound1 == self.sound_names[i]:
                train1 = self.Fouriers[40 * i:40 * i + 35]
            if sound2 == self.sound_names[i]:
                train2 = self.Fouriers[40 * i:40 * i + 35]

        try:
            if 'train1' not in locals():
                raise NameError('sound1 was not found.')
            if 'train2' not in locals():
                raise NameError('sound2 was not found.')
        except NameError as message:
            print(message)

        mixture = list()
        ideal_mask = list()
        correct = list()
        while True:
            first = train1[self.train_num[0]]
            second = train2[self.train_num[1]]

            try:
                if first.shape[-1] < 100:
                    raise TypeError("The data is too short.")
            except TypeError as message:
                print(message)

            if first.shape[-1] > 100:
                if first.shape[-1] < (self.iterate_num + 1) * 100:
                    first = np.concatenate((first[:, self.iterate_num * 100:],
                                            first[:, :(100 - (first.shape[-1] - self.iterate_num * 100))]), axis=1)
                    second = np.concatenate((second[:, self.iterate_num * 100:],
                                             second[:, :(100 - (second.shape[-1] - self.iterate_num * 100))]), axis=1)

                    self.train_num[1] += 1
                    if len(train2) == self.train_num[1]:
                        self.train_num[0] += 1
                        self.train_num[1] = 0
                        if len(train1) == self.train_num[0]:
                            self.train_num[0] = 0
                    self.iterate_num = 0
                else:
                    first = first[:, self.iterate_num * 100:(self.iterate_num + 1) * 100]
                    second = second[:, self.iterate_num * 100:(self.iterate_num + 1) * 100]
                    self.iterate_num += 1
            else:
                self.train_num[1] += 1
                if len(train2) == self.train_num[1]:
                    self.train_num[0] += 1
                    self.train_num[1] = 0
                    if len(train1) == self.train_num[0]:
                        self.train_num[0] = 0

            with np.errstate(all="raise"):
                mix = first + second
                mix = np.log(np.abs(mix) + 0.0001)

                first = np.log(np.abs(first) + 0.0001)
                second = np.log(np.abs(second) + 0.0001)

                mixture.append(mix)
                correct.append(np.transpose(np.asarray([first, second]), axes=[1, 2, 0]))
                max = np.max(np.asarray([first, second]), axis=0)
                first = np.logical_not(first - max).astype(np.float32)
                second = np.logical_not(second - max).astype(np.float32)
                if not np.any(first):
                    # print('first is zeros')
                    mixture.pop()
                    correct.pop()
                    continue
                if not np.any(second):
                    # print('second is zeros')
                    mixture.pop()
                    correct.pop()
                    continue
                ideal_mask.append([first, second])

                if len(mixture) == batch_size:
                    x_train1 = np.asarray(mixture)
                    x_train2 = np.asarray(ideal_mask)
                    y_train = np.asarray(correct)

                    mixture = list()
                    ideal_mask = list()
                    correct = list()

                    yield [x_train1, x_train2], y_train

    def get_batch_and_steps(self, model):
        batch_size = model.get_batch_size()
        steps = (((self.Fouriers.shape[-1]//100)+1)*35*35) // batch_size
        return batch_size, steps