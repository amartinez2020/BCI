#library imports
import joblib
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from ast import literal_eval
from psychopy import gui


import re
import os
from time import sleep

from scipy import signal as sg
import pyeeg
from pywt import wavedec, downcoef, wavedec2


from mne.filter import filter_data
from mne.decoding import CSP

from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline

def bandpass_filter(data,fs=500,low_freq=7,high_freq=30,method='fir'):
    """
    Apply band pass filter to eeg data

    Parameters
    ----------

    data: ndarray
        Signal data to filter
    fs: int
        Sampling frequency
    low_freq: int
        Lower frequency band
    high_freq: int
        Higher frequency band

    Returns
    ----------
    filtered_data: ndarray
        Signal data constrained to given frequency bands
    """
    filtered_data = filter_data(data,fs,low_freq,high_freq,method=method)
    return(filtered_data)



def extract_band(data,fs,band='alpha'):
    """
    Extract frequency band from eeg data

    Parameters
    ----------

    data: ndarray
        Signal data to filter

    fs: int
        Sampling frequency

    band: str
        Frequency band to extract (delta, theta, alpha, beta, gamma)


    Returns
    ----------
    filtered_data: ndarray
        Signal data constrained to given band
    """
    bands = {'delta':(None,4),'theta':(4,8),'alpha':(8,12),'beta':(12,30),'gamma':(30,None)}
    low_freq = bands[band][0]
    high_freq = bands[band][1]
    filtered_data = filter_data(data,fs,low_freq,high_freq)
    return(filtered_data)

def get_samples(trial_times,stim_times,df_path,eeg_df_path,erp_cushion=0):
    """
    Fills experiment dataframe with corresponding eeg data per trial

    Parameters
    ----------
    trial_times : dict
        A dictionary that maps each mental task trial to a timestamp window

    stim_times : dict
        A dictionary that maps each stimulus presentation to a timestamp window

    df_path : str
        Path to experiment dataframe

    eeg_df_path : str
        Path to eeg dataframe

    erp_cushion : int
        Time to record (in seconds) before and after onset of mental task
    """

    #load dfs from paths

    eeg_df = pd.read_csv(eeg_df_path)
    df = pd.read_csv(df_path)

    #cast experiment df to object type
    df = df.astype(object)

    #iterate through every trial and extract eeg samples
    for (trial_key,trial_value),(stim_key,stim_value)  in zip(trial_times.items(),stim_times.items()):

        print(f'experiment run and trial: {trial_key}')

        #onset and offset times
        print(f'trial start and stop: {trial_value}')
        print(f'stimulus start and stop: {stim_value}')

        #extract eeg data between onset and offset times
        trial_data = eeg_df[eeg_df['Local Clock'].between(trial_value[0]-erp_cushion, trial_value[1]+erp_cushion)]['EEG Sample']
        stim_data = eeg_df[eeg_df['Local Clock'].between(stim_value[0]-erp_cushion, stim_value[1]+erp_cushion)]['EEG Sample']


        print(f'duration (s): {trial_value[1]-trial_value[0]}')

        #set trial and stim data values to samples variable
        trial_samples = trial_data.values
        stim_samples = stim_data.values

        #display number of samples per trial
        print(f'length of samples: {len(trial_samples)}')
        print("\n")

        #trial_info & stim_info
        trial_info = {'key': list(trial_key),'samples':trial_samples}
        stim_info = {'key': list(stim_key),'samples': stim_samples}

        #insert experiment df with eeg data
        insert_raw_samples(df,trial_info,stim_info,df_path)


    #save df
    df.to_csv(df_path,index=False)


def insert_raw_samples(df,trial_info,stim_info,df_path):
    """
    Inserts eeg data into experiment dataframe

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Experiment dataframe

    trial_info : dict
        Dictionary containing trial start and stop times, and eeg samples

    stim_info : dict
        Dictionary containing stimulus start and stop times, and eeg samples

    df_path : str
        Path to experiment dataframe
    """

    #extract run and trial information and store in variables
    run = trial_info['key'][0]
    trial = trial_info['key'][1]

    #extract index from experiment df for which to insert eeg data
    index = int(df.loc[df['Run']==run][df['Trial']==trial].index[0])

    #format trial samples from strings to list of floats
    formatted_trial_samples = []
    for sample in trial_info['samples']:
        formatted_trial_samples.append(literal_eval(sample))

    #format stim samples from strings to list of floats
    formatted_stim_samples = []
    for sample in stim_info['samples']:
        formatted_stim_samples.append(literal_eval(sample))

    #insert formatted samples in experiment df

    df.loc[index,'Trial EEG Samples'],df.loc[index,'Stim EEG Samples'] = formatted_trial_samples, formatted_stim_samples


def get_stim_times(logfile):
    """
    Extracts stimulus times from experiment logfile

    Parameters
    ----------
    logfile : logfile
        Experiment logfile

    Returns
    ----------
    stim_times : dict
        A dictionary that maps each stimulus presentation to a timestamp window
    """

    #regular expressions used to extract data from logfile
    eeg_regex = 'eeg.*'
    stim_start_regex = 'stim.*autoDraw = True'
    stim_end_regex = 'stim.*autoDraw = False'
    run_regex = 'break.*autoDraw = True'

    #return variable
    stim_times = {}

    #variable that stores timestamp when eeg was started
    eeg_start = None

    #open logfile and extract stimulus presentation times
    with open(logfile) as file:
        lines = file.read().splitlines()
        stim_start = stim_end = trial_count = run_count = 0
        for line in lines:

            #format each line as 3 element array
            line = line.split()
            if len(line) > 0:
                line[2] = " ".join(line[2:])
                line = line[:3]

                #if current line is eeg start time, then log its start time
                if re.compile(eeg_regex).match(line[2]):
                    eeg_start = float(line[2].split()[-1])-float(line[0])

                #if stimulus starts
                elif re.compile(stim_start_regex).match(line[2]):
                    stim_start = float(line[0])+eeg_start

                #if stim ends
                elif re.compile(stim_end_regex).match(line[2]):
                    stim_end = float(line[0])+eeg_start
                    stim_times[(run_count,trial_count)]=[stim_start,stim_end]
                    trial_count += 1

                #if experiment run ended
                elif re.compile(run_regex).match(line[2]):
                    run_count += 1

    return(stim_times)

def get_subject_info(header):
    '''
    input: text to show at top of pop-up (string)
           path to data directory (string)

    Creates pop up box to obtain subject# and run#
    '''
    info = {}
    info['Subject #'] = ''
    dlg = gui.DlgFromDict(dictionary=info, title=header)
    if dlg.OK:
        return("subject_" + str(info['Subject #']))
    else:
        print("Error!")

def get_times(logfile):
    """
    Extracts mental task trial times from experiment logfile

    Parameters
    ----------
    logfile : logfile
        Experiment logfile

    Returns
    ----------
    trial_times : dict
        A dictionary that maps each mental task trial to a timestamp window
    """
    #regular expressions used to extract data from logfile
    eeg_regex = 'eeg.*'
    stim_regex = 'stim.*autoDraw = False'
    data_regex = 'fix.*autoDraw = False'
    run_regex = 'break.*autoDraw = True'

    #return variable
    trial_times = {}

    #variable that stores timestamp when eeg was started
    eeg_start = None

    #open logfile and extract mental task trial presentation times
    with open(logfile) as file:
        lines = file.read().splitlines()
        last = load_start = load_end = None
        start = False
        trial_count = run_count = 0
        for line in lines:

            #format each line as 3 element array
            line = line.split()
            if len(line) > 0:
                line[2] = " ".join(line[2:])
                line = line[:3]

                #if current line is eeg start time, then log its start time
                if re.compile(eeg_regex).match(line[2]):
                    eeg_start = float(line[2].split()[-1])-float(line[0])

                #if experiment run ended
                elif re.compile(run_regex).match(line[2]):
                    run_count += 1

                #find load end time
                elif start and re.compile(data_regex).match(line[2]):
                    load_end = float(line[0])+eeg_start
                    trial_times[(run_count,trial_count)]=[load_start,load_end]
                    trial_count += 1
                    start = False

                elif last:
                    #if stim fixation just ended then this current event is the start of our eeg load
                    if re.compile(stim_regex).match(last[2]):
                        load_start = float(line[0])+eeg_start
                        start = True

                last = line

    return(trial_times)


def plot_signal(signal_data,name='Signal'):
    x = np.linspace(0,7, num=len(list(signal_data)))
    #eeg_data[0][0]


    fig, ax = plt.subplots()
    ax.plot(x,signal_data)
    plt.ylabel('Voltage (mV)')
    plt.xlabel('Time (S)')
    ax.set_title(name)


def plot_trial(df_path,subject_info,run,trial,bandpass=False,method=None,dwt=False,sub_band=None,psd=False,save=True,name_extension=None):
    """
    Creates ERP plots for each electrode for a given run and trial

    Parameters
    ----------
    df_path : str
        Path to experiment dataframe

    subject_info : str
        Subject identification number

    run : int
        Run identification number

    trial : int
        Trial identification number
    """




    #load experiment df
    df = pd.read_csv(df_path)

    #extract eeg samples for this specific trial
    index = int(df.loc[df['Run']==run][df['Trial']==trial]['Trial EEG Samples'].index[0])
    samples = literal_eval(df.loc[index,'Trial EEG Samples'])



    #hex color codes for each electrode
    colors = ['#F0A3FF', '#0075DC', '#993F00', '#4C005C', '#191919', '#005C31', '#2BCE48', '#FFCC99', '#808080', '#94FFB5', '#8F7C00', '#9DCC00', '#C20088', '#003380', '#FFA405', '#FFA8BB', '#426600', '#FF0010', '#5EF1F2', '#00998F', '#E0FF66', '#740AFF', '#990000', '#FFFF80', '#FFFF00', '#FF5005']

    #create list of dictionaries containing x (timestamp), y(voltage), color, and width values for each electrode
    all_signals = []
    for each_electrode in range(20):
        signal = f'Electrode{each_electrode}'
        all_signals.append({'name':signal,'x':[],'y':[],'color':colors[each_electrode],'linewidth':1})

    # #iterate through all eeg samples
    for sample in samples:
        #separate out timestamps and voltage values from eeg samples
        all_readings = sample[0][:20]
        timestamp = sample[1]

        #for each electrode update the all_signals dictionary
        for each_electrode in range(len(all_readings)):
            all_signals[each_electrode]['y'].append(all_readings[each_electrode])
            all_signals[each_electrode]['x'].append(timestamp)



    #create subplots
    for signal in range(len(all_signals)):

        fig, ax = plt.subplots()

        #normalize timestamps to start from zero
        first_timestamp = all_signals[signal]['x'][0]
        for each_timestamp in range(len(all_signals[signal]['x'])):
            all_signals[signal]['x'][each_timestamp] = all_signals[signal]['x'][each_timestamp] - first_timestamp

        #apply filters
        if bandpass:
            x,y = all_signals[signal]['x'], bandpass_filter(all_signals[signal]['y'],fs=500,low_freq=7,high_freq=30,method=method)

        elif sub_band:
            x,y = all_signals[signal]['x'], extract_band(all_signals[signal]['y'],fs=500,band=sub_band)
        elif dwt:
            # coeffs = wavedec(all_signals[signal]['y'], dwt)[-1]
            coeffs = downcoef('d', all_signals[signal]['y'], 'coif1', level=6)
            x,y = np.linspace(0,all_signals[signal]['x'][-1],num=len(coeffs)),coeffs
        else:
            x,y = all_signals[signal]['x'], all_signals[signal]['y']



        #plot each electrode
        if psd:

            plt.psd(y,Fs=500)

            #format plot
            ax.set_axisbelow(True)
            ax.minorticks_on()
            ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
            ax.set_title(f"PSD for Run: {run} Trial: {trial} Electrode: {signal}")
            plt.ylabel('Power Spectral Density (dB/Hz)')
            plt.xlabel('Frequency')
            # plt.xlim(0, 150)
            # plt.ylim(-15,15)
        else:
            ax.plot(x,y,
                color=all_signals[signal]['color'],
                linewidth=all_signals[signal]['linewidth'],
                label=all_signals[signal]['name'])

            #format plot
            ax.set_title(f"ERP for Run: {run} Trial: {trial} Electrode: {signal}")
            plt.ylabel('Voltage (mV)')
            plt.xlabel('Time (S)')

        #save plot
        if save:
            if name_extension:
                plt.savefig(f'data/{subject_info}/ERP_plots/run{run}_trial{trial}_electrode{signal}_{name_extension}.png')
            else:
                plt.savefig(f'data/{subject_info}/ERP_plots/run{run}_trial{trial}_electrode{signal}.png')
        else:
            plt.show()



def train(subject_info,classes,classifiers,model_name,transforms,k=5,n_components=4):
    """
    Train classifier on mental imagery classes

    Parameters
    ----------
    subject_info : str
        Subject identification number

    classes : list
        Mental imagery classes to be separated

    classifiers : dict
        Maps model name to randomly initialized binary classification models

    model_name : str
        Binary classification model name

    k: int
        Hyperparameter for k-fold cross validation

    n_components: int
        Number of common spatial feature components to extract from signals
    """


    #binary label encoding
    encoding = {classes[0]:0, classes[1]:1}



    #load subject data as df
    df = pd.read_csv(os.path.join(os.getcwd(),f'{subject_info}.csv'))

    #filter df by class
    filter_df = df[df['Category'].str.contains(classes[0]) | df['Category'].str.contains(classes[1])]

    #set X and y values
    X = filter_df['Feature Vector']
    y = [encoding[x] for x in list(filter_df['Category'])]

    #split data into train test sets
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=1)

    #get classifier
    model = classifiers[model_name]

    #set save path
    save_path = os.path.join(os.getcwd(),f'{model_name}_{classes[0]}_{classes[1]}.sav')


    #create pipeline with transform and model
    steps = transforms.append((model_name, model))
    clf = Pipeline(steps)
    print(len(X_train),len(y_train))
    #get cross validation score
    scores = cross_val_score(clf, X_train, y_train, cv=k, n_jobs=1)
    print(scores.mean())

    #extract common spatial features
    X_train, X_test = csp.fit_transform(X_train, y_train), csp.transform(X_test)

    #fit to train data and save best model
    model.fit(X_train,y_train,verbose=1,max_iter=60)

    #score model on validation data
    validation_score = model.score(X_test,y_test)
    print(validation_score)

    #save model
    joblib.dump(model,save_path)
