#library imports
from psychopy import visual, core, event,logging, gui
from PIL import Image
import os
import sys
import pandas as pd
import random
from pylsl import StreamInlet, resolve_stream, local_clock
from multiprocessing import Process, Queue, freeze_support,set_executable
import threading
import time
import pickle
import numpy as np

# # first resolve an EEG stream on the lab network
# print("looking for an EEG stream...")
# streams = resolve_stream('type', 'EEG')
#
# # create a new inlet to read from the eeg stream
# inlet = StreamInlet(streams[0])

def load_data(subject_info,paths,df,q):
    '''
    Loads data from eeg into dataframe during thread call

    Parameters
    ----------

    subject_info : str
        Subject identification number

    paths : dict
        A dictionary that maps paths to subject data files and experiment stimili

    df : pandas.core.frame.DataFrame
        A dataframe that stores continuous eeg sampling data

    q : queue
        A queue object used to exit out of load data
        If empty, eeg sampling continues indefinitely
    '''

    print("running thread")

    #index of eeg df
    index = 0

    #as long as queueu is empty thread will run continuously
    while q.empty():
        df.loc[index] = [subject_info,[inlet.pull_sample()][0][1],[inlet.pull_sample()][0]]
        index += 1
    print("halting thread")

    #save eeg data to df
    df.to_csv(paths['data_path']+"/"+subject_info+'_eeg.csv')
    #print('load end time: ' + str(clock.getTime()))

def fix_stim(win):
    """
    Creates fixation stimulus

    Parameters
    ----------
    win : psychopy.visual.window.Window
        Psychopy visual window

    Returns
    ----------
    stim : psychopy.visual.text.TextStim
        Central fixation stimulus for display in window
    """

    #create fixation stimulus
    stim = visual.TextStim(win=win, ori=0, name='fixation_cross', text='+', font='Arial',
                  height = 2, color='lightGrey', colorSpace='rgb', opacity=1, depth=0.0)
    return(stim)

def break_stimulus(win,break_stim):
    """
    Displays break stimulus in between experiment runs

    Parameters
    ----------
    win : psychopy.visual.window.Window
        Psychopy visual window

    break_stim : psychopy.visual.text.TextStim
        Experiment run break stimulus
    """
    #start core clock
    clock = core.Clock()

    #while space bar is not pressed continue to show break stimulus
    #if 50 seconds pass, then quit experiment
    break_stim.setAutoDraw(True)
    while not event.getKeys(['space']):
        win.flip()
        if int(clock.getTime()) > 50:
            core.quit
    break_stim.setAutoDraw(False)

def end_stimulus(win,end_stim):
    """
    Displays end stimulus at the end of the experiment

    Parameters
    ----------
    win : psychopy.visual.window.Window
        Psychopy visual window

    end_stim : psychopy.visual.text.TextStim
        Experiment end stimulus
    """
    #start core clock
    clock = core.Clock()

    #while space bar is not pressed continue to show end stimulus
    #if 50 seconds pass, then stop showing end stimulus
    end_stim.setAutoDraw(True)
    while not event.getKeys(['space']):
        win.flip()
        if int(clock.getTime()) > 50:
            break
    end_stim.setAutoDraw(False)

def as_stim(text,win):
    return(visual.TextStim(win=win, ori=0, name="stimulus_text", text=text, font='Arial',height = 2, color='lightGrey', colorSpace='rgb', opacity=1, depth=0.0))


def experiment_run(subject_info,win,cwd,fixation,paths,rest,stim_list,timing,df,break_stim,end_stim,test=False):
    """
    Main experiment presentation function

    Parameters
    ----------
    subject_info : str
        Subject identification number

    win : psychopy.visual.window.Window
        Psychopy visual window

    cwd : str
        Current working directory

    fixation : psychopy.visual.text.TextStim
        Central fixation stimulus for display in window

    paths : dict
        A dictionary that maps paths to subject data files and experiment stimili

    rest : psychopy.visual.text.TextStim
        Rest stimulus shown in between trial runs

    stim_list : dict
        A dictionary that maps stimulus names to image files

    timing : dict
        A dictionary that maps stimuli to their presentation times

    df : pandas.core.frame.DataFrame
        A dataframe that stores all experiment data

    break_stim : psychopy.visual.text.TextStim
        Break stimulus shown in between experiment runs

    end_stim : psychopy.visual.text.TextStim
        End stimulus shown at the end of the experiment

    test : bool
        If true, run test experiment, else run train experiment
    """
    #run variable
    current_run = 0
    if test:
        #iterate through dataframe
        for index, row in df.iterrows():
            #if we encounter a new experiment run then display the break stimulus and update run variable
            if df.at[index,'Run']!=current_run:
                break_stimulus(win,break_stim)
                current_run = df.at[index,'Run']

            #create psychopy image stimulus from image in df
            img = df.at[index,'Image']
            img1,img2 = Image.open(img[0]).convert('LA'), Image.open(img[1]).convert('LA')
            img1,img2 = visual.ImageStim(win, img1, size=(10),pos=(-10, 0.0), name='stimulus_image1'), visual.ImageStim(win, img2, size=(10), pos=(10, 0.0), name='stimulus_image2')

            #initialize clock
            clock = core.Clock()

            #start fixation stimulus
            fixation.setAutoDraw(True)
            while clock.getTime()<timing['fix']:
                win.flip()
            fixation.setAutoDraw(False)

            #imagery task stimulus
            img1.setAutoDraw(True)
            img2.setAutoDraw(True)
            while clock.getTime()<timing['stim']:
                win.flip()
            img1.setAutoDraw(False)
            img2.setAutoDraw(False)

            #end fixation stimulus
            fixation.setAutoDraw(True)
            while clock.getTime()<timing['trial']:
                win.flip()
            fixation.setAutoDraw(False)

            #display rest stimulus for three seconds
            rest.draw()
            win.flip()
            core.wait(3)

            #save df
            df.to_csv(paths['data_path']+"/"+subject_info+'.csv')

    else:

        #iterate through dataframe
        for index, row in df.iterrows():
            #if we encounter a new experiment run then display the break stimulus and update run variable
            if df.at[index,'Run']!=current_run:
                break_stimulus(win,break_stim)
                current_run = df.at[index,'Run']

            #create psychopy image stimulus from image in df
            img = df.at[index,'Image']
            image = Image.open(os.path.join(paths['stim_path'],img)).convert('LA')
            img = visual.ImageStim(win, image, size=(10), name='stimulus_image')

            #initialize clock
            clock = core.Clock()

            #start fixation stimulus
            fixation.setAutoDraw(True)
            while clock.getTime()<timing['fix']:
                win.flip()
            fixation.setAutoDraw(False)

            #imagery task stimulus
            img.setAutoDraw(True)
            while clock.getTime()<timing['stim']:
                win.flip()
            img.setAutoDraw(False)

            #end fixation stimulus
            fixation.setAutoDraw(True)
            while clock.getTime()<timing['trial']:
                win.flip()
            fixation.setAutoDraw(False)

            #display rest stimulus for three seconds
            rest.draw()
            win.flip()
            core.wait(3)

            #save df
            df.to_csv(paths['data_path']+"/"+subject_info+'.csv')

    #end experiment
    end_stimulus(win,end_stim)


def init_df(subject_info,paths,params,stim_list,train=True,eeg=False):
    """
    Initialize dataframes

    Parameters
    ----------
    subject_info : str
        Subject identification number

    paths : dict
        A dictionary that maps paths to subject data files and experiment stimili

    params : dict
        A dictionary that maps runs and trials to their respective lengths

    stim_list : dict
        A dictionary that maps stimulus names to image files

    eeg : bool
        If True, initialize eeg dataframe, else initialize experiment dataframe

    Returns
    ----------
    df : pandas.core.frame.DataFrame
        A dataframe containing either experiment or eeg data
    """

    if eeg==False:
        #columns for experiment df
        columns = ['Subject','Run','Trial', 'Category','Image','Trial EEG Samples','Stim EEG Samples','Feature Vector']
        df = pd.DataFrame(index=(range(params['runs']*params['trials_per_run'])),columns=columns,dtype=object)
        #fill in values
        df['Subject'] = subject_info
        df['Run'],df['Trial'] = trial_setup(params)
        print(len(stim_generate(params,stim_list,train=train)[0]),len(stim_generate(params,stim_list,train=train)[1]))
        df['Category'],df['Image'] = stim_generate(params,stim_list,train=train)

        #save df
        df.to_csv(paths['data_path']+"/"+subject_info+'.csv')
    else:
        #columns for eeg df
        columns = ['Subject', 'Local Clock', 'EEG Sample']
        df = pd.DataFrame(columns=columns,dtype=object)
        #fill in subject info
        df['Subject'] = subject_info
        #save df
        df.to_csv(paths['data_path']+"/"+subject_info+'_eeg.csv')
    return(df)

def pre_questionnaire(subject_info, paths=None, save=False):
    '''
    Create pop up box to obtain and save subject's demographic info

    input:    info - dictionary containing participant# and run#
              save - boolean indicating whether to autosave
              save_path - if save==True, path to data save location

    output:   if save==True, save out data, return nothing
              if save==False, return questionnaire data
    '''

    preDlg = gui.Dlg()

    preDlg.addField('1. Age')
    preDlg.addText('')
    preDlg.addField('2. Sex:')
    preDlg.addText('')
    preDlg.addField('3. Are you hispanic or latino?', choices=['--', "Yes", "No"])
    preDlg.addText('4. Race (check all that apply):')
    preDlg.addField('White', False)
    preDlg.addField('Black or African American', False)
    preDlg.addField('Native Hawaiian or other Pacific Islander', False)
    preDlg.addField('Asian', False)
    preDlg.addField('American Indian or Alaskan Native', False)
    preDlg.addField('Other', False)
    preDlg.addField('Prefer to not disclose', False)
    preDlg.addText('')
    preDlg.addField('5. Highest Degree Achieved:', choices = ['--', 'some high school', 'high schoool graduate', 'some college', \
    'college graduate', 'some graduate training', "Master's", 'Doctorate'])
    preDlg.addText('')
    preDlg.addField('6. What is your birth country?')
    preDlg.addText('')
    preDlg.addField('7. Do you have normal color vision?', choices = ['--',"Yes", "No"])
    preDlg.addText('')
    preDlg.addField('8. Are you taking any medications or have you had any recent injuries that could affect your attention?',choices = ['--', "Yes", "No"])
    preDlg.addField('9. If yes to question above, describe')
    preDlg.addText('')
    preDlg.addField('10. How many hours of sleep did you get last night?')
    preDlg.addText('')
    preDlg.addField('11. How many cups of coffee have you had today?')
    preDlg.addText('')
    preDlg.addField('12. How alert are you feeling?:', choices=['--', "Not alert", "Neutral", "Very alert"])

    end_data = preDlg.show()

    if save:
        name = paths['data_path'] + '/pre_questionnaire_' + subject_info + '.pkl'
        with open(name, 'wb') as f:
            pickle.dump(end_data, f)
    else:
        return(end_data)

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

def trial_setup(params):
    """
    Setup runs and trials

    Parameters
    ----------
    params : dict
        A dictionary that maps runs and trials to their respective lengths

    Returns
    ----------
    runs : list
        A list containing run indices

    trials : list
        A list containing trial indices
    """
    runs = []
    trials = []
    for run in range(params['runs']):
        runs = runs + [run]*params['trials_per_run']
        for trial in range(params['trials_per_run']):
            trials.append(trial)
    return(runs,trials)


def stim_generate(params,stim_list,train):
    """
    Randomly assign stimuli to trials for each run

    Parameters
    ----------
    params : dict
        A dictionary that maps runs and trials to their respective lengths

    stim_list : dict
        A dictionary that maps stimulus names to image files

    Returns
    ----------
    shuffled_stim : list
        A list containing image categories for each trial

    shuffled_images : list
        A list containing image files for each trial
    """
    if train:
        stim = list(stim_list.keys())
        shuffled_stim = shuffled_images = []
        #for each run get equal amounts of stim and shuffle
        #can only work if 'trials_per_run' is divisible by number of stims
        for run in range(params['runs']):
            temp_list = stim * int(params['trials_per_run']/len(stim))
            random.shuffle(temp_list)
            shuffled_stim = shuffled_stim + temp_list
        for stim in shuffled_stim:
            shuffled_images.append(stim_list[stim])
    else:
        #open the test stim images
        images = []
        for each in os.listdir(stim_list):
            if '.DS_Store' not in each:
                img = os.path.join(stim_list,each)
                images.append([img,img])

        #randomize images
        random.shuffle(images)

        #split images into matches and not matches
        matches,not_matches = images[:len(images)//2],images[len(images)//2:]

        #shift the image list so that the pictures dont match
        not_matches = shift_list(not_matches)

        #combine matches and not matches and randomize order
        shuffled_images = matches + not_matches
        random.shuffle(shuffled_images)

        #get category labels
        shuffled_stim = []
        for img in shuffled_images:
            if img[0] == img[1]:
                shuffled_stim.append("match")
            else:
                shuffled_stim.append("not match")

    return(shuffled_stim,shuffled_images)


def shift_list(old_list):
    i = 0
    j = i + 1
    new_list = []
    while i < len(old_list):
        if j == len(old_list):
            new_list.append([old_list[i][0],old_list[0][1]])
        else:
            new_list.append([old_list[i][0],old_list[j][1]])
        i += 1
        j += 1
    return(new_list)
