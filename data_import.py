import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import h5py
import pickle
import matplotlib.pyplot as plt
from scipy.stats import entropy

def import_flows(file, input_if=None):
    samples = pd.read_csv(file).sort_values('unixSecondsUTC')
    samples = samples.drop_duplicates()

    if input_if:
        samples = samples[samples.inputIF == input_if]

    return samples


def import_counters(file, if_name=None):
    counters = pd.read_csv(file).sort_values('unixSecondsUTC')
    counters = counters.drop_duplicates()

    if if_name:
        counters = counters[counters.ifName == if_name]

    return counters

def cal_entropy(x):
    unique, counts = np.unique(x, return_counts=True)
    return entropy(np.divide(counts, len(x)))

def generate_detectors(counter_df, flow_df):
    counter_df = counter_df[counter_df.unixSecondsUTC < flow_df.iloc[len(flow_df) - 1]['unixSecondsUTC']]
    ##Group by 30 sec time slot, this timestamp given in counters csv
    bin = counter_df['unixSecondsUTC']
    groups = flow_df.groupby(pd.cut(flow_df.unixSecondsUTC, bin))
    avg_ip_size = groups.IPSize.mean().values
    out_octet = counter_df.ifInOctets.diff().values[1:]
    no_of_flows = groups.size().values
    byte_per_flow = np.zeros(len(no_of_flows))
    byte_per_flow[no_of_flows != 0] = out_octet[no_of_flows != 0] / no_of_flows[no_of_flows != 0]
    entropy_by_grou p = groups.dstIP.agg(cal_entropy)
    # pof_dstIP_bygroup = groups.dstIP.value_counts()/no_of_flows_by_group


    time_stamps = counter_df['unixSecondsUTC'].values[1:]

    return {
        'time_stamps': time_stamps,
        'avg_ip_size': avg_ip_size,
        'no_of_flows': no_of_flows,
        'Byte_count': out_octet,
        'entropy': entropy_by_group.values,
        'byte_per_flow': byte_per_flow
    }

#
# def plot(deterctor_dict):
#     fig = 1
#     for type in deterctor_dict:
#         if type is not 'time_stamps':
#             plt.figure(fig)
#             fig += 1
#             plt.plot(deterctor_dict[type], 'ro')
#             plt.ylabel(type)
#
#     plt.show()

def prepare_data(data_dict, columns, scaler=None):
    # column is a ist of keys in the data_dict
    data = np.empty((0, len(columns)))

    first = True
    for col in columns:
        if first:
            data = np.array(data_dict[col])
            first = False
        else:
            data = np.column_stack((data, data_dict[col]))

    # Get rid of Nan
    for c in range(0, data.shape[1]):
        data = data[~np.isnan(data[:, c])]

    if not scaler:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(data)

    data = scaler.transform(data)

    return scaler, data


def analyse_data(flow_file, cnt_file, ifce=None):

    if ifce:
        return generate_detectors(import_counters(cnt_file, ifce['name']),
                                  import_flows(flow_file, ifce['id']))

    else:
        return generate_detectors(import_counters(cnt_file),
                                  import_flows(flow_file))



def make_supervised_learn_problem(data, seq_length, no_of_predictions):

    total_length = seq_length + no_of_predictions

    ##Walk forward by one by one
    shaped_data = np.empty((len(data) - total_length, total_length, data.shape[1]))
    current_index = 0

    for i in range(total_length, len(data)):
        shaped_data[current_index] = np.reshape(data[current_index:current_index + total_length], (1, total_length, data.shape[1]))
        current_index = current_index + 1

    #shaped_data.shape = (no.of_trainexamples, total_length, np_of_features)

    x = shaped_data[:, :-no_of_predictions, :]
    y = shaped_data[:, -no_of_predictions, :]

    return x,y


if __name__ == '__main__':

#############TRAINING
    #no_attack_param = analyse_data('Data/Normal/sflow_FLOW.csv', 'Data/Normal/sflow_CNT.csv', {'name': 's1-eth1', 'id': 2342})
    no_attack_param = analyse_data('Data/Infocom/Normal/32hrs/sflow_FLOW.csv', 'Data/Infocom/Normal/32hrs/sflow_CNT.csv',
                                   {'name': 's1-eth1', 'id': 2300})

    scaler_val, prep_data = prepare_data(no_attack_param, ['no_of_flows', 'Byte_count', 'avg_ip_size'])
    split_pt = round(len(prep_data) * 2/3)
    x_train, y_train = make_supervised_learn_problem(prep_data[:split_pt], 119, 1)
    x_val, y_val = make_supervised_learn_problem(prep_data[split_pt+1:], 119, 1)
    #
    ########SAVE TRAINING DATA
    with h5py.File('Data/Infocom/Normal/data.h5', 'w') as hf:
        hf.create_dataset("x_train", data=x_train)
        hf.create_dataset("y_train", data=y_train)
        hf.create_dataset("x_val", data=x_val)
        hf.create_dataset("y_val", data=y_val)

    ####Save Scaler

    scaler_pkl= open('Data/Infocom/Normal/scaler.pkl', 'wb')
    pickle.dump(scaler_val, scaler_pkl)
    scaler_pkl.close()

    ###Save raw data
    data_file = open('Data/Infocom/Normal/raw_data.pkl', 'wb')
    pickle.dump(no_attack_param, data_file)
    data_file.close()


########ATTACK
    attack_param = analyse_data('Data/Infocom/Attack/sflow_FLOW.csv', 'Data/Infocom/Attack/sflow_CNT.csv',
                                        {'name': 's1-eth1', 'id': 2292})

    scaler_test, prep_data = prepare_data(attack_param, ['no_of_flows', 'Byte_count', 'avg_ip_size'], scaler_val)

    x_test, y_test = make_supervised_learn_problem(prep_data, 119, 1)
    #
    ########SAVE DATA
    with h5py.File('Data/Infocom/Attack/data.h5', 'w') as hf:
        hf.create_dataset("x_test", data=x_test)
        hf.create_dataset("y_test", data=y_test)

    # ####Save Scaler
    #
    # scaler_pkl = open('Data/Infocom/Attack/scaler.pkl', 'wb')
    # pickle.dump(scaler_test, scaler_pkl)
    # scaler_pkl.close()

    ##Save Raw Data
    test_data_file = open('Data/Infocom/Attack/raw_data.pkl', 'wb')
    pickle.dump(attack_param, test_data_file)
    data_file.close()



    # plt.figure(1)
    # plt.plot(no_attack_param['byte_per_flow'], 'r-', attack_param['byte_per_flow'], 'b-')
    #
    # plt.figure(2)
    # plt.plot(no_attack_param['avg_ip_size'], 'r-', attack_param['avg_ip_size'], 'b-')
    #
    # plt.show()