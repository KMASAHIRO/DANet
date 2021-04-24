import tensorflow as tf
import itertools
import soundfile as sf
import pandas
import numpy as np
import scipy
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.style as ms
ms.use('seaborn-muted')

# 前処理をするレイヤ(推論時にモデルが位相付きのデータを受け取り、Maskとかけ合わせるために必要)
class Preparation(tf.keras.layers.Layer):
  def __init__(self, log_eps,  *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.log_eps = log_eps

  def call(self, input, training):
    if training:
      return input
    else:
      model_input = tf.math.log(tf.math.abs(input) + self.log_eps)
      return model_input

# Attractorを生成するレイヤ(基本的に学習時にはideal mask、推論時にはkmeansを使用)
class Attractor(tf.keras.layers.Layer):
    def __init__(self, kmeans_func, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kmeans_func = kmeans_func
        self.is_kmeans = False

    def call(self, input, training):
        if training:
            if self.is_kmeans:
                attractor = self.kmeans_func(input[0])
                attractor = tf.convert_to_tensor(attractor)
            else:
                att_num = tf.einsum('Ncft,Nftk->Nck', input[0], input[1])
                att_denom = tf.math.reduce_sum(input[0], axis=[2, 3])  # batch_size, c
                att_denom = tf.reshape(att_denom, [-1, 2, 1])
                attractor = att_num / att_denom
        else:
            attractor = self.kmeans_func(input[0])
            attractor = tf.convert_to_tensor(attractor)

        return attractor


# Maskと混合音声を掛け合わせて分離音声を生成するレイヤ
class Make_clean_reference(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def call(self, input, training):
    if training:
      clean_reference = tf.einsum('Nft,Nftc->Nftc',input[0],input[1])
      return clean_reference
    else:
      clean_reference = tf.einsum('Nft,Nftc->Nftc',tf.cast(input[0], dtype=tf.complex64),tf.cast(input[1], dtype=tf.complex64))
      return clean_reference


# DANetのモデル
class DANet(tf.keras.Model):
    def __init__(self, source_num, embed_ndim, batch_size, log_eps=0.0001, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.source_num = source_num
        self.embed_ndim = embed_ndim
        self.log_eps = log_eps
        self.batch_size = batch_size
        self.cluster_centers_list = np.ones(shape=(self.batch_size, self.source_num, self.embed_ndim))

        self.preparation = Preparation(self.log_eps)
        self.reshape = tf.keras.layers.Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1]))
        self.lstm1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(300, return_sequences=True),
                                                   merge_mode='concat')
        self.lstm2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(300, return_sequences=True),
                                                   merge_mode='concat')
        self.lstm3 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(300, return_sequences=True),
                                                   merge_mode='concat')
        self.lstm4 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(300, return_sequences=True),
                                                   merge_mode='concat')
        self.embedding1 = tf.keras.layers.Dense(129 * self.embed_ndim)
        self.embedding2 = tf.keras.layers.Reshape((100, 129, self.embed_ndim))
        self.embedding3 = tf.keras.layers.Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1, 3]))
        self.make_attractor = Attractor(self.kmeans_predict)
        self.make_mask = tf.keras.layers.Lambda(lambda x: tf.einsum('Nftk,Nck->Nftc', x[0], x[1]))
        self.make_clean_reference = Make_clean_reference()

    def call(self, inputs, training):
        x1 = self.preparation(inputs[0], training)
        x1 = self.reshape(x1)
        x1 = self.lstm1(x1)
        x1 = self.lstm2(x1)
        x1 = self.lstm3(x1)
        x1 = self.lstm4(x1)
        x1 = self.embedding1(x1)
        x1 = self.embedding2(x1)
        x1 = self.embedding3(x1)
        attractor = self.make_attractor([inputs[1], x1], training)
        mask = tf.keras.activations.softmax(self.make_mask([x1, attractor]))
        clean_reference = self.make_clean_reference([inputs[0], mask], training)

        return clean_reference

    def train_with_kmeans(self, generator, steps, epochs, ideal_epochs):
        loss_result = list()
        for epoch in range(epochs):
            loss_epoch = list()
            if epoch == ideal_epochs:
                self.to_kmeans_train()
            for step in range(steps):
                train_x, train_y = next(generator)
                if ideal_epochs <= epoch:
                    self.kmeans_fit(train_x[0])
                loss = self.train_on_batch(x=train_x, y=train_y)
                loss_epoch.append(loss)
            print("Epoch {}/{}".format(epoch + 1, epochs), "loss: {:.2f}".format(np.mean(loss_epoch)), sep=" ",
                  flush=True)
            loss_result.append(np.mean(loss_epoch))
        return loss_result

    def to_kmeans_train(self):
        self.make_attractor.is_kmeans = True

    def to_idealmask_train(self):
        self.make_attractor.is_kmeans = False

    # embeddingの結果を得る関数(kmeansクラスタリングを行うときに使用)
    def get_embedded_data(self, inputs, training):
        x1 = self.preparation(inputs, training)
        x1 = self.reshape(x1)
        x1 = self.lstm1(x1)
        x1 = self.lstm2(x1)
        x1 = self.lstm3(x1)
        x1 = self.lstm4(x1)
        x1 = self.embedding1(x1)
        x1 = self.embedding2(x1)
        output = self.embedding3(x1)

        return output

    # kmeansクラスタリングを行う関数
    def kmeans_fit(self, inputs, max_iter=1000, random_seed=0):
        embedded_data = self.get_embedded_data(inputs, training=False)

        shape = embedded_data.shape
        embedded_data = np.reshape(embedded_data, newshape=(shape[0], shape[1] * shape[2], shape[3]))

        cluster_centers_list = list()

        for n in range(len(inputs)):
            X = embedded_data[n]
            random_state = np.random.RandomState(random_seed)

            cycle = itertools.cycle(range(self.source_num))
            labels = np.fromiter(itertools.islice(cycle, X.shape[0]), dtype=np.int)
            random_state.shuffle(labels)
            labels_prev = np.zeros(X.shape[0])
            cluster_centers = np.zeros((self.source_num, X.shape[1]))

            for i in range(max_iter):
                for k in range(self.source_num):
                    XX = X[labels == k, :]
                    cluster_centers[k, :] = XX.mean(axis=0)

                dist = ((X[:, :, np.newaxis] - cluster_centers.T[np.newaxis, :, :]) ** 2).sum(axis=1)
                labels_prev = labels
                labels = dist.argmin(axis=1)

                for k in range(self.source_num):
                    if not np.any(labels == k):
                        labels[np.random.choice(len(labels), 1)] = k

                if (labels == labels_prev).all():
                    break

            for k in range(self.source_num):
                XX = X[labels == k, :]
                cluster_centers[k, :] = XX.mean(axis=0)

            cluster_centers_list.append(cluster_centers)

        self.cluster_centers_list = np.asarray(cluster_centers_list)

    # kmeansクラスタリングの結果(中心点)を受け取る関数
    def kmeans_predict(self, input):
        return self.cluster_centers_list

    # 設定したbatch sizeを得る関数(計算グラフ構築時、モデルへの入力のbatch sizeとこの値が一致していなければならない)
    def get_batch_size(self):
        return self.batch_size

    # batch sizeを設定する関数(計算グラフ構築時、モデルへの入力のbatch sizeとこの値が一致していなければならない)
    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.cluster_centers_list = np.ones(shape=(self.batch_size, self.source_num, self.embed_ndim))

    # 推論を行う関数
    def prediction(self, input):
        self.kmeans_fit(input)
        fake_ideal_mask = np.zeros(shape=(input.shape[0], self.source_num, 129, 100))
        result = self.predict([input, fake_ideal_mask], batch_size=len(input))
        return result

    # モデルの重みをロードする関数
    def loading(self, path):
        input1 = np.zeros(shape=(self.batch_size, 129, 100))
        input2 = np.zeros(shape=(self.batch_size, self.source_num, 129, 100))
        temp = self.predict(x=[input1, input2], batch_size=self.batch_size)
        self.load_weights(path)

# 損失関数
def loss_function(y_true, y_pred):
  frequency = tf.shape(y_true)[1]
  time = tf.shape(y_true)[2]
  frequency = tf.cast(frequency, tf.float32)
  time = tf.cast(time, tf.float32)
  return tf.reduce_sum((y_true - y_pred)**2) / (frequency*time)

# モデルを構築する関数
def create_model(source_num=2, embed_ndim=20, optimizer=None, loss=loss_function):
    batch_size = 25
    model = DANet(source_num=source_num, embed_ndim=embed_ndim, batch_size=batch_size, log_eps=0.0001)

    if optimizer is None:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.0001, decay_steps=51450,
                                                                     decay_rate=0.03)
        model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=lr_schedule), loss=loss)
    else:
        model.compile(optimizer=optimizer, loss=loss)

    input1 = np.zeros(shape=(batch_size, 129, 100))
    input2 = np.zeros(shape=(batch_size, source_num, 129, 100))
    temp = model.predict(x=[input1, input2], batch_size=batch_size)

    return model