
from data_import import analyse_data, prepare_data
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
import h5py

# no_attack_param = analyse_data('Data/Normal/32hrs/sflow_FLOW.csv', 'Data/Normal/32hrs/sflow_CNT.csv', {'name': 's1-eth1', 'id':2288})



def build_lstm(input_dim, batch_size, lstm_layers, dropout=0.0, recurrent_dropout=0.0):
    model = Sequential()
    model.add(LSTM(lstm_layers[0], batch_input_shape=(batch_size, input_dim[0], input_dim[1]),return_sequences=True, stateful=True, dropout=dropout, recurrent_dropout=recurrent_dropout))

    for i in range(1, len(lstm_layers) - 2):
        model.add(LSTM(lstm_layers[i], return_sequences=True, stateful=True, dropout=dropout, recurrent_dropout=recurrent_dropout))

    model.add(LSTM(lstm_layers[len(lstm_layers) - 1], return_sequences=False, stateful=True, dropout=dropout, recurrent_dropout=recurrent_dropout))
    model.add(Dense(input_dim[1])) ## No activation means liniear activation
    model.compile(loss='mean_squared_error', optimizer='rmsprop')

    return model


# fit an LSTM network to training data
def build_and_train_model(x, y, batch_size, nb_epoch, lstm_layers, validation_data=None, dropout=0.0, recurrent_dropout=0.0):

    model = build_lstm((x.shape[1], x.shape[2]), batch_size, lstm_layers, dropout=dropout, recurrent_dropout=recurrent_dropout)
    p_model = build_lstm((x.shape[1], x.shape[2]), 1, lstm_layers)

    loss = np.empty((nb_epoch,2))

    print("build_and_train_model")
    for i in range(nb_epoch):
        print("in epoch" + str(i))
        h = model.fit(x, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.save_weights('lstm_model.h5')
        p_model.load_weights('lstm_model.h5')
        loss[i, 0] = h.history['loss'][0]
        # loss[i, 1] = h.history['val_loss'][0]
        loss[i, 1] = predict_and_get_mse(p_model, validation_data[0], validation_data[1])
        model.reset_states()
    return loss, model


# one-step forecast for entire array
def predict_and_get_mse(model, X, Y):
    Y_hat = np.empty((len(X),Y.shape[1]))
    for x in range(len(X)):
        Y_hat[x] = model.predict(np.array([X[x,:,:]]), batch_size=1)

    return mean_squared_error(Y, Y_hat)


if __name__ == '__main__':
    # no_attack_param = analyse_data('Data/Normal/32hrs/sflow_FLOW.csv', 'Data/Normal/32hrs/sflow_CNT.csv',
    #                                {'name': 's1-eth1', 'id': 2300})
    #
    # scaler, prep_data = prepare_data(no_attack_param, ['byte_per_flow', 'avg_ip_size'])
    # split_pt = round(len(prep_data) * 2/3)
    # x_train, y_train = make_supervised_learn_problem(prep_data[:split_pt], 119, 1)
    # x_val, y_val = make_supervised_learn_problem(prep_data[split_pt+1:], 119, 1)

    #########SAVE DATA
    # with h5py.File('Data/Normal/32hrs/data.h5', 'w') as hf:
    #     hf.create_dataset("x_train", data=x_train)
    #     hf.create_dataset("y_train", data=y_train)
    #     hf.create_dataset("x_val", data=x_val)
    #     hf.create_dataset("y_val", data=y_val)

    #######LOAD DATA
    with h5py.File('Data/Normal/32hrs/data.h5', 'r') as hf:
        x_train = hf['x_train'][:]
        y_train = hf['y_train'][:]
        x_val = hf['x_val'][:]
        y_val = hf['y_val'][:]

    # print(x_val.shape)
    #
    loss, model = build_and_train_model(x_train, y_train, 40, 40, [50, 50], validation_data=(x_val,y_val), recurrent_dropout=0.3)
    model.save('my_model.h5')
    x = np.array(range(0, len(loss)))
    plt.plot(x, loss[:,0], 'r--', x, loss[:,1], 'b--')
    plt.show()

    print("done")



