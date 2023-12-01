import pandas as pd
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from veremi.config import Config


def load_veremi(csv_file: str, feature: str, label: str, delimiter=','):
    # Import VeReMi Dataset
    data = pd.read_csv(csv_file, delimiter=delimiter)

    # select columns
    columns = []
    for column in data.columns.values:
        if feature == 'feat1':
            if 'RSSI' in column:
                columns.append(column)
            elif 'distance' in column:
                columns.append(column)
        elif feature == 'feat2':
            if 'conformity' in column and '0' not in column:
                columns.append(column)
        elif feature == 'feat3':
            if 'RSSI' in column and '0' not in column:
                columns.append(column)
            elif 'distance' in column and '0' not in column:
                columns.append(column)
            elif 'conformity' in column and '0' not in column:
                columns.append(column)
        elif feature == 'feat4':
            if 'RSSI' in column:
                columns.append(column)
            elif 'aoa' in column:
                columns.append(column)
            elif 'distance' in column:
                columns.append(column)
            elif 'conformity' in column and '0' not in column:
                columns.append(column)
    columns.append('attackerType')

    # process target values
    pos_label = 1
    if label == 'multiclass':
        data = data[columns]
    elif label == 'binary':
        data = data[columns]
        data['attackerType'].loc[data['attackerType'] != 0] = pos_label
    else:
        pos_label = int(label.split("_")[1])
        data = data[columns]
        data = data.loc[(data['attackerType'] == 0) | (data['attackerType'] == pos_label)]

    data_normal = data.loc[data['attackerType'] == 0]
    data_atk = data.loc[data['attackerType'] != 0]
    # atk_size = int(data_atk.shape[0] * 1.5)
    atk_size = int(data_atk.shape[0])
    data = pd.concat([data_normal.sample(atk_size), data_atk])
    data = shuffle(data)

    dataset = data
    target = data[data.columns[-1:]]
    data = data[data.columns[0:-1]]

    # normalize data
    data = (data - data.mean()) / data.std()

    # label binarize one-hot style
    lb = preprocessing.LabelBinarizer()
    lb.fit(target)
    if label == 'multiclass':
        target = lb.transform(target)
    else:
        target = lb.transform(target)
        target = MultiLabelBinarizer().fit_transform(target)

    # Create training and test data
    train_data, test_data, train_labels, test_labels = train_test_split(
        data,
        target,
        train_size=Config.data_train_size,
        test_size=Config.data_test_size,
        # random_state=42
    )

    return train_data, test_data, train_labels, test_labels, lb, dataset
