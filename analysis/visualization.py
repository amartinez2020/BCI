#library imports
from analysis_helpers import *
import os
from time import sleep

#experiment title
title = "BCI"

#current path
os.chdir("..")
cwd = os.getcwd()
# print(cwd)

#get subject info
# subject_info = get_subject_info(title)
subject_info = "subject_50"
#paths
logfile = os.path.join(cwd,f'data/{subject_info}/{subject_info}_log.log')
eeg_df = os.path.join(cwd,f'data/{subject_info}/{subject_info}_eeg.csv')
experiment_df = os.path.join(cwd,f'data/{subject_info}/{subject_info}.csv')
# experiment_df = f'data/{subject_info}/{subject_info}.csv'

#subband frequencies
sub_bands = {'delta','theta','alpha','beta','gamma'}

#get filter from user input
print("\n")
print("Signal Transformation: dwt, bandpass, {'delta','theta','alpha','beta','gamma'}, or None")
input_filter = input()
if input_filter == 'bandpass':
    print("Method: fir or iir")
    method = input()



#power spectral density
print("Power Spectral Density: true or false")
psd = input()
if psd == "true": psd = True
else: psd = False

#get desired trials
print("Enter Trial: int (x) if single trial, else type 'r' ")
trial = input()
if trial == 'r':
    print("Start trial")
    start_trial = int(input())
    print("Stop trial")
    stop_trial = int(input())
    trial = [start_trial,stop_trial]
else:
    trial = int(trial)


#save plots
print("Save plots? (true or false)")
save = input()
if save == "true": save = True
else: save = False



#create directory if save is True
if save == True:
    try:
        os.mkdir(f'data/{subject_info}/ERP_plots')
    except:
        print()

#plot indicated trials
if input_filter == 'dwt':

    if isinstance(trial,list):

        for each_trial in range(trial[0],trial[1]):

            plot_trial(experiment_df,subject_info,0,each_trial,dwt='coif1',psd=psd,save=save)
    else:
        print(type(trial))
        plot_trial(experiment_df,subject_info,0,trial,dwt='coif1',psd=psd,save=save)
elif input_filter == 'bandpass':
    if isinstance(trial,list):
        for each_trial in range(trial[0],trial[1]):
            plot_trial(experiment_df,subject_info,0,each_trial,bandpass=True,method=method,psd=psd,save=save)
    else:
        plot_trial(experiment_df,subject_info,0,trial,bandpass=True,method=method,psd=psd,save=save)

elif input_filter in sub_bands:
    if isinstance(trial,list):
        for each_trial in range(trial[0],trial[1]):

            plot_trial(experiment_df,subject_info,0,each_trial,sub_band=input_filter,psd=psd,save=save)
    else:
        plot_trial(experiment_df,subject_info,0,trial,sub_band=input_filter,psd=psd,save=save)
else:
    if isinstance(trial,list):
        for each_trial in range(trial[0],trial[1]):
            plot_trial(experiment_df,subject_info,0,each_trial,psd=psd,save=save)
    else:
        plot_trial(experiment_df,subject_info,0,trial,psd=psd,save=save)
