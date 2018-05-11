from keras.models import load_model
from data_import import analyse_data, prepare_data
import numpy as np
from lstm import build_lstm
import h5py
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
from sklearn.preprocessing import MinMaxScaler
import pickle


def generate_error(model, X, Y, scaler):
    Y_hat = np.empty((len(X), Y.shape[1]))

    for x in range(len(X)):
        Y_hat[x] = model.predict(np.array([X[x,:,:]]), batch_size=1)

    Y_unscaled = scaler.inverse_transform(Y)
    Y_hat_unscaled = scaler.inverse_transform(Y_hat)
    return  Y_unscaled - Y_hat_unscaled


if __name__ == '__main__':

    ######Load Data

    ##VALidation
    with h5py.File('Data/Infocom/Normal/data.h5', 'r') as hf:
        x_val = hf['x_val'][:]
        y_val = hf['y_val'][:]

    with open('Data/Infocom/Normal/scaler.pkl', 'rb') as pkfile:
        scaler = pickle.load(pkfile)

    ###TEST
    with h5py.File('Data/Infocom/Attack/data.h5', 'r') as hf:
        x_test = hf['x_test'][:]
        y_test = hf['y_test'][:]

    # with open('Data/Attack/scaler.pkl', 'rb') as pkfile:
    #     scaler_test = pickle.load(pkfile)


    model = load_model('Data/Infocom/my_model.h5')
    p_model = build_lstm((x_val.shape[1], x_val.shape[2]), 1, [50, 50])
    model.save_weights('Data/Infocom/weights.h5')
    p_model.load_weights('Data/Infocom/weights.h5')

    errors = generate_error(p_model, x_val, y_val, scaler)
    # plt.figure(1)
    # plt.hist(errors[:,0], 50, normed=1, facecolor='green', alpha=0.5)
    # plt.figure(2)
    # plt.hist(errors[:, 1], 50, normed=1, facecolor='blue', alpha=0.5)
    # plt.show()

    mean = np.mean(errors, axis=0)
    var = np.var(errors, axis=0)
    cov = np.cov(errors, rowvar=False)

    mvn = multivariate_normal(mean=mean, cov=cov, allow_singular = True)
    # nd = [norm(mean[i], var[i]) for i in range(len(mean))] ##array of gaussin distru=ibut for each feature

    errors_p = generate_error(p_model, x_test, y_test, scaler)

    plt.figure(1)

    plt.plot(np.log10(mvn.pdf(errors)), 'r-')
    plt.figure(2)
    plt.plot(np.log10(mvn.pdf(errors_p)))

    plt.show()


    print("Done")
    # scipy.stats.gaussian_kde(errors())