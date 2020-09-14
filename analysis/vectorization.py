#library imports
from analysis_helpers import *
import os
from time import sleep

#experiment title
title = "BCI"

#current path
os.chdir("..")
cwd = os.getcwd()

#get subject info
subject_info = get_subject_info(title)

#paths
logfile = os.path.join(cwd,f'data/{subject_info}/{subject_info}_log.log')
eeg_df = os.path.join(cwd,f'data/{subject_info}/{subject_info}_eeg.csv')
experiment_df = os.path.join(cwd,f'data/{subject_info}/{subject_info}.csv')

#load trial times
trial_times = get_times(logfile)
stim_times = get_stim_times(logfile)

#load experiment df with eeg samples
get_samples(trial_times,stim_times,experiment_df,eeg_df,erp_cushion=1)
