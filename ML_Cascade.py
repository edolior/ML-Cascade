import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import metrics
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import *


"""
Welcome to the Cascade Meta-Learner Project.
Machine Learning Course Assignment No. 2 - BGU SISE 2021.
"""


def predict_cascade_model(x_test, y_test, threshold_conf, l_targets, l_classifiers):
    """
    function preprocesses datasets and inserts them to a cascade prediction model
    :param x_test and y_test are the tests sets
    :param threshold_conf 0.95 constant
    :param l_targets list of multi-classifiers
    :param l_classifiers list of decision tree classifiers by a changing number of depth
    :return results of the cascade prediction model including accuracy and log loss
    """
    count, correct = 0, 0
    l_true, l_pred_probs = [], []
    d_depths = {}
    np_depths = np.zeros(15)
    for index, row in x_test.iterrows():
        i = 0
        while i < 15:
            curr_pred_prob = l_classifiers[i].predict_proba([row])[0]
            y_pred = l_classifiers[i].predict([row])
            curr_max = max(curr_pred_prob)
            if curr_max > threshold_conf or i == 14:
                if y_test[index] == y_pred:
                    correct += 1
                    l_true.append(y_test[index])
                l_pred_probs.append(curr_pred_prob)
                try:
                    np_depths[i] += 1
                    d_depths[i] += 1
                except KeyError as e:
                    d_depths[i] = 1
                break
            i += 1
        count += 1
    plot_model_depths(d_depths)
    cascade_accuracy = correct/count
    curr_pred_prob = l_pred_probs
    f_log_loss = log_loss(y_test, curr_pred_prob, labels=l_targets)
    return cascade_accuracy, f_log_loss


def plot_model_depths(d_depths):
    d_depths = dict(sorted(d_depths.items()))
    cmap = plt.cm.tab10
    colors = cmap(np.arange(len(d_depths)) % cmap.N)
    plt.bar(*zip(*d_depths.items()), color=colors)
    plt.show()
    plt.draw()


def set_one_hot_vector(l_cols, df_dummies):
    """
    function creates a one-hot vector
    :param l_cols list of columns to transform (must be strings)
    :param df_dummies columns to apply
    """
    df_one_hot = pd.DataFrame()
    for col in l_cols:
        curr_list = (df_dummies[col])
        curr_dummies = pd.get_dummies(curr_list, prefix=[col])
        df_one_hot = df_one_hot.append(curr_dummies)
    return df_one_hot


def target_apply(curr_row, s_target):
    """
    function creates a pre-defined one hot vector using pre-defined settings
    :param curr_row current value in the dataframe
    :param s_target column to apply
    """
    value = curr_row.iloc[0][s_target].astype(float)
    if value >= 0.75:
        return 'High'
    elif 0.25 <= value < 0.75:
        return 'Medium'
    elif value < 0.25:
        return 'Low'
    else:
        return ''


def set_file_list(p_resources):
    """
    function loads datasets
    :param p_resources path to datasets directory.
    :return dictionary of datasets
    """
    d_curr = {}
    for root, dirs, files in os.walk(p_resources):
        for file in files:
            curr_file_path = os.path.join(root, file)
            i_delimeter = curr_file_path.rfind('\\')
            s_filename = curr_file_path[i_delimeter+1:]
            d_curr[s_filename] = 0
    return d_curr


def data_report(df_data, l_iterations, b_df):
    """
    function shows data distribution of how many values of features and classes exists per dataset
    :param df_data dataframe to examine
    :param l_iterations classes to calculate missing values
    :param b_df boolean flag True for dataframe and False for series type
    :return plot
    """
    d_percentages = {}
    if b_df:
        for curr_col in df_data[l_iterations[:10]]:
            i_null = ~df_data[curr_col].isnull()
            i_null_all = float(i_null.sum())
            f_perc = i_null_all / df_data.shape[0] * 100
            d_percentages[curr_col] = float(f_perc)
            # print('%s, Is Missing: %d (%.2f%%)' % (curr_col, i_null_all, perc))
    else:
        df_data = df_data.to_frame(name='Class')
        for curr_class in l_iterations:
            f_perc = df_data[df_data['Class'] == curr_class].value_counts().values[0]
            f_perc /= df_data.shape[0]
            f_perc *= 100
            d_percentages[curr_class] = f_perc
    cmap = plt.cm.tab10
    colors = cmap(np.arange(len(df_data)) % cmap.N)
    plt.bar(*zip(*d_percentages.items()), color=colors)
    plt.show()
    plt.draw()


def dataset_builder(d_datasets, s_max_features, s_criterion, s_splitter, b_random, s_exp):
    """
    function preprocesses datasets and inserts them to a cascade prediction model
    :param p_resources path directory containing 10 datasets
    :return results of the cascade prediction model including accuracy and log loss
    """
    MAX_LIMIT = 50000
    for s_data, value in d_datasets.items():
        x_data = None
        y_data = None
        p_curr_data = p_resources + '\\' + s_data
        if 'connect-4' in p_curr_data:
            curr_df = pd.read_csv(p_curr_data, delimiter=",")
            curr_df = curr_df.replace("b", 1)
            curr_df = curr_df.replace("x", 0)
            curr_df = curr_df.replace("o", 2)
            if curr_df.shape[0] > MAX_LIMIT:
                curr_df = curr_df[:MAX_LIMIT]
            y_data = curr_df['win']
            x_data = curr_df.drop("win", axis=1)
        elif 'avila' in p_curr_data:
            curr_df = pd.read_csv(p_curr_data)
            if curr_df.shape[0] > MAX_LIMIT:
                curr_df = curr_df[:MAX_LIMIT]
            y_data = curr_df['Class']
            x_data = curr_df.drop("Class", axis=1)
        elif 'letter-recognition' in p_curr_data:
            curr_df = pd.read_csv(p_curr_data)
            if curr_df.shape[0] > MAX_LIMIT:
                curr_df = curr_df[:MAX_LIMIT]
            y_data = curr_df['T']
            x_data = curr_df.drop("T", axis=1)
        elif 'segmentation' in p_curr_data:
            curr_df = pd.read_csv(p_curr_data, skiprows=2, error_bad_lines=False, delimiter=",")
            if curr_df.shape[0] > MAX_LIMIT:
                curr_df = curr_df[:MAX_LIMIT]
            curr_df.reset_index(level=0, inplace=True)
            y_data = curr_df['index']
            x_data = curr_df.drop("index", axis=1)
        elif 'log2' in p_curr_data:
            curr_df = pd.read_csv(p_curr_data)
            if curr_df.shape[0] > MAX_LIMIT:
                curr_df = curr_df[:MAX_LIMIT]
            y_data = curr_df['Action']
            x_data = curr_df.drop("Action", axis=1)
        elif 'Dry_Bean_Dataset' in p_curr_data:
            curr_df = pd.read_csv(p_curr_data)
            if curr_df.shape[0] > MAX_LIMIT:
                curr_df = curr_df[:MAX_LIMIT]
            y_data = curr_df['Class']
            x_data = curr_df.drop("Class", axis=1)
        elif 'ObesityDataSet_raw_and_data_sinthetic' in p_curr_data:
            curr_df = pd.read_csv(p_curr_data)
            if curr_df.shape[0] > MAX_LIMIT:
                curr_df = curr_df[:MAX_LIMIT]
            y_data = curr_df['NObeyesdad']
            x_data = curr_df.drop("NObeyesdad", axis=1)
            x_one_hot = set_one_hot_vector(["Gender", "CAEC", "CALC", "MTRANS"], x_data)
            print('Data Size: ', x_one_hot.shape)
            x_data = x_data.replace("yes", 1)
            x_data = x_data.replace("no", 0)
            x_data = x_data.replace("Male", 1)
            x_data = x_data.replace("Female", 0)
            x_data = x_data.drop(["CAEC", "CALC", "MTRANS"], axis=1)
        elif 'pendigits' in p_curr_data:
            curr_df = pd.read_csv(p_curr_data)
            if curr_df.shape[0] > MAX_LIMIT:
                curr_df = curr_df[:MAX_LIMIT]
            s_target = 'ans'
            y_data = curr_df[s_target]
            y_one_hot = curr_df.groupby(s_target).apply(target_apply, s_target)
            print('Data Size: ', y_one_hot.shape)
            x_data = curr_df.drop(s_target, axis=1)
        elif 'training' in p_curr_data:
            curr_df = pd.read_csv(p_curr_data)
            if curr_df.shape[0] > MAX_LIMIT:
                curr_df = curr_df[:MAX_LIMIT]
            y_data = curr_df['class']
            x_data = curr_df.drop("class", axis=1)
        elif 'sensor_readings' in p_curr_data:
            curr_df = pd.read_csv(p_curr_data)
            if curr_df.shape[0] > MAX_LIMIT:
                curr_df = curr_df[:MAX_LIMIT]
            s_target = 'Class'
            y_data = curr_df[s_target]
            x_data = curr_df.drop(s_target, axis=1)

        l_targets = y_data.unique()
        if len(l_targets) > 0:
            value = l_targets[0]
            if isinstance(value, str):
                l_targets = sorted(l_targets, key=str.lower)
            else:
                l_targets = sorted(l_targets)
        d_datasets[s_data] = l_targets
        l_features = list(x_data.columns)
        data_report(x_data, l_features, True)
        data_report(y_data, l_targets, False)

        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)
        if not b_random:
            curr_x_train, curr_x_test, curr_y_train, curr_y_test = x_train, x_test, y_train, y_test
        threshold_conf = 0.95  # requested constant parameters
        threshold_depth = 16
        l_classifiers = list()  # list of classifiers of ascending numbers of depths

        j = 1
        while j < threshold_depth:  # loads cascade models
            if b_random:
                curr_x_train, curr_x_test, curr_y_train, curr_y_test = train_test_split(x_train, y_train, test_size=0.2)
            clf_i = DecisionTreeClassifier(max_depth=j, max_features=s_max_features, splitter=s_splitter,
                                           criterion=s_criterion)
            clf_i = clf_i.fit(curr_x_train, curr_y_train)
            l_classifiers.append(clf_i)
            j += 1

        print('Running Training on File: ' + s_data)
        cascade_accuracy, f_cascade_log_loss = predict_cascade_model(x_test, y_test, threshold_conf, l_targets,
                                                                     l_classifiers)

        print(s_exp + " Cascade Model Accuracy: " + "{:.3f}".format(cascade_accuracy))
        print(s_exp + " Cascade Model Log Loss: " + "{:.3f}".format(f_cascade_log_loss))


# (0) SET PATHS TO YOUR PROJECT DIRECTORY & LOADS DATASETS #
p_project = os.path.dirname(os.path.dirname(__file__))
p_src = p_project + r'\ML_Cascade'
p_resources = p_src + r'\resources'
if not os.path.exists(p_resources):
    os.makedirs(p_resources)
d_datasets = set_file_list(p_resources)


# (1) CASCADE MODEL - ORIGINAL #
s_exp = 'Original'
s_max_features = None
s_criterion = "entropy"  # based on information gain
s_splitter = "best"
b_random = 'False'
dataset_builder(d_datasets, s_max_features, s_criterion, s_splitter, b_random, s_exp)


# (2) CASCADE MODEL - RANDOM #
s_exp = 'Shuffled'
s_max_features = None
s_criterion = "entropy"
s_splitter = "random"
b_random = 'True'
dataset_builder(d_datasets, s_max_features, s_criterion, s_splitter, b_random, s_exp)


# (3) CASCADE MODEL - FEATURE SELECTION #
s_exp = 'Split Feature Selection'
s_max_features = "sqrt"
s_criterion = "entropy"
s_splitter = "best"
b_random = 'False'
dataset_builder(d_datasets, s_max_features, s_criterion, s_splitter, b_random, s_exp)
