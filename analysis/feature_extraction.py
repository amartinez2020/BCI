#library imports
from analysis_helpers import *
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score

from mne import Epochs, pick_types, events_from_annotations
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP

#experiment title
title = "BCI"

#current path
os.chdir("..")
cwd = os.getcwd()
print(cwd)

#get subject info
# subject_info = get_subject_info(title)
subject_info = "subject_50"

#paths
logfile = os.path.join(cwd,f'data/{subject_info}/{subject_info}_log.log')
eeg_df = os.path.join(cwd,f'data/{subject_info}/{subject_info}_eeg.csv')
experiment_df = os.path.join(cwd,f'data/{subject_info}/{subject_info}.csv')
experiment_df = f'data/{subject_info}/{subject_info}.csv'



#apply band pass filters to data
df = pd.read_csv(experiment_df)


#per trial
for index,row in df.iterrows():
    samples = literal_eval(df.loc[index,'Trial EEG Samples'])

    #electrode array is of shape (20,~1875)
    electrode_array = []
    timestamps = []

    for sample in samples:
        #separate out timestamps and voltage values from eeg samples
        electrode_array.append(sample[0][:20])
        timestamps.append(sample[1])













#
#
######## Authors: Martin Billinger <martin.billinger@tugraz.at> ########
tmin, tmax = -1, 4
event_id = dict(hands=2, feet=3)
subject = 1
runs = [6, 10, 14]  # motor imagery: hands vs feet

raw_fnames = eegbci.load_data(subject, runs)
raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
eegbci.standardize(raw)  # set channel names
montage = make_standard_montage('standard_1005')
raw.set_montage(montage)

# strip channel names of "." characters
raw.rename_channels(lambda x: x.strip('.'))

# Apply band-pass filter

raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')

events, _ = events_from_annotations(raw, event_id=dict(T1=2, T2=3))

picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                   exclude='bads')

# Read epochs (train will be done only between 1 and 2s)
# Testing will be done with a running classifier
epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                baseline=None, preload=True)

epochs_train = epochs.copy().crop(tmin=1., tmax=2.)
labels = epochs.events[:, -1] - 2
######## Authors: Martin Billinger <martin.billinger@tugraz.at> ########
# #visualize transformed data
# # epoch_number = 1
# # for epoch in epochs.get_data()[:3]:
# #     #show one signal at a time
# #     for channel in epoch[:5]:
# #         plt.plot(channel)
# #
# #     #plt.savefig(os.path.join(os.getcwd(),f'mne_data: epoch{epoch_number}'))
# #     plt.show()
# #     epoch_number += 1
# #
# # #apply hilbert transform
# # new_epochs = epochs.apply_hilbert()
# #
# # epoch_number = 1
# # for epoch in new_epochs.get_data()[:3]:
# #     #show one signal at a time
# #     for channel in epoch[:5]:
# #         plt.plot(channel)
# #
# #     #plt.savefig(os.path.join(os.getcwd(),f'mne_data: epoch{epoch_number}'))
# #     plt.show()
# #     epoch_number += 1
scores = []
epochs_data = epochs.get_data()
epochs_data_train = epochs_train.get_data()
cv = ShuffleSplit(10, test_size=0.2, random_state=42)
cv_split = cv.split(epochs_data_train)

# Assemble a classifier
lda = LinearDiscriminantAnalysis()
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

# Use scikit-learn Pipeline with cross_val_score function
clf = Pipeline([('CSP', csp), ('LDA', lda)])
scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=1)

# Printing the results
class_balance = np.mean(labels == labels[0])
class_balance = max(class_balance, 1. - class_balance)
print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores),
                                                          class_balance))

# plot CSP patterns estimated on full data for visualization
csp.fit_transform(epochs_data, labels)
print(csp.patterns_.shape)

csp.plot_patterns(epochs.info, ch_type='eeg', units='Patterns (AU)', size=1.5)
csp.plot_filters(epochs.info, ch_type='eeg', units='Patterns (AU)', size=1.5)
