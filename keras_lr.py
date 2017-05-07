from time import time
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


import keras
from keras.datasets import mnist
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.losses import categorical_crossentropy
from keras.models import Model
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras import backend as K
from keras.regularizers import l2
from skopt import dump
from skopt import gp_minimize
from skopt import forest_minimize
from skopt.callbacks import TimerCallback

(X, y), _ = mnist.load_data()
X = X.astype('float32')
X /= 255.0
X_train, X_val, y_train, y_val = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=0)
y_train = LabelBinarizer().fit_transform(y_train)
y_val = LabelBinarizer().fit_transform(y_val)
n_classes = y_train.shape[1]

X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))

def lr_objective(x):
    t = time()
    lr, l2_reg, batch_size, epochs = x
    reg = l2(l=l2_reg)
    sgd = SGD(lr=lr, clipnorm=1.0)

    input_ = Input(shape=(X_train.shape[1],))
    dense = Dense(n_classes, kernel_regularizer=reg, activation="sigmoid")(input_)
    #output = Dropout(rate=dropout)(dense)
    model = Model(inputs=input_, outputs=dense)
    model.compile(
        loss=categorical_crossentropy,
        optimizer=sgd,
        metrics=["accuracy"])
    model.fit(
        X_train, y_train, batch_size=batch_size, verbose=1, epochs=1)
    score = model.evaluate(X_val, y_val, verbose=0)
    if np.any(np.isnan(score)):
        return 1.0
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return 1 - score[1]

bounds = [[10**-3, 10**0.0, "log-uniform"],
          [10**-3.0, 10**0.0, "log-uniform"],
          [12, 126],
          [1, 100]]

def run(optimizer, n_calls, n_runs):
    if optimizer == "gp":
        opt = gp_minimize
    elif optimizer == "forest":
        opt = forest_minimize

    min_vals = []
    all_vals = []
    all_times = []
    for n_run in range(n_runs):
        timer = TimerCallback()
        res = opt(
            lr_objective, bounds, n_calls=n_calls, n_random_starts=1,
            verbose=1, random_state=n_run, callback=timer)
        del res["specs"]
        dump(res, "%s_%d.pkl" % (optimizer, n_run))

        fun_pkl = open("%s_%d_fun.pkl" % (optimizer, n_run), "wb")
        pickle.dump(res.fun, fun_pkl)
        fun_pkl.close()

        yi_pkl = open("%s_%d_yi.pkl" % (optimizer, n_run), "wb")
        pickle.dump(res.func_vals, yi_pkl)
        yi_pkl.close()

        time_pkl = open("%s_%d_times.pkl" % (optimizer, n_run), "wb")
        pickle.dump(timer.iter_time, time_pkl)
        time_pkl.close()

        min_vals.append(res.fun)
        all_vals.append(res.func_vals)
        all_times.append(timer.iter_time)

    print(min_vals)
    print(all_vals)
    print(all_times)
    return np.min(min_vals), np.std(min_vals)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--optimizer', nargs="?", default="gp", type=str, help="gp | forest")
    parser.add_argument(
        '--n_calls', nargs="?", default="50", type=int, help="Number of calls")
    parser.add_argument(
        '--n_runs', nargs="?", default="5", type=int, help="Number of runs")
    args = parser.parse_args()
    min_vals, std_vals = run(args.optimizer, args.n_calls, args.n_runs)
    print(min_vals)
    print(std_vals)
