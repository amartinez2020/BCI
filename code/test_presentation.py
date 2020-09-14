#library imports
from psychopy.preferences import prefs
from presentation_helpers import *

#experiment title
title = "BCI"

#quit key
event.globalKeys.add(key='q', func=core.quit, name='shutdown')

#current path
os.chdir("..")
cwd = os.getcwd()

#get subject info
ext = "_test"
subject_info = get_subject_info(title) + ext


#paths
paths = {'test_stim_path':os.path.join(cwd,'test_stim'),'data_path':os.path.join(cwd,'data/'+str(subject_info[:-len(ext)]))}


#logging
logging.setDefaultClock(core.Clock())
logging.LogFile(f=(paths['data_path']+"/"+subject_info+"_log.log"),filemode='w',level=logging.DEBUG)
logging.log(level=logging.WARN, msg='warning')
logging.log(level=logging.EXP , msg='experiment')
logging.log(level=logging.DATA, msg='data')
logging.log(level=logging.INFO, msg='info')
logging.info("eeg time: " + str(local_clock()))

#params
params = {'runs':5,'trials_per_run':20}
#for testing ends of runs
#params = {'runs':1,'trials_per_run':6}

#window and frame rate variables
win = visual.Window([1680,1050], fullscr = False, monitor = 'testMonitor', units='deg', color = 'black')
win.logOnFlip(level=logging.EXP, msg='frame flip')
rate = win.getActualFrameRate()

#timing
timing = {'fix':3, 'stim':4.5, 'trial':10}

#fixation cross
fixation = fix_stim(win)

#rest stimulus
rest = visual.TextStim(win,text='REST',height=2)

#break and end stims
break_stim = visual.TextStim(win,text='End of run. Press Spacebar to continue.', name='break_stimulus', height=1)
end_stim = visual.TextStim(win,text='End of Experiment! Press Spacebar to exit.', name='end_stimulus',height=1)

#initialize dfs
df = init_df(subject_info,paths,params,paths['test_stim_path'],train=False)
eeg_df = init_df(subject_info,paths,params,paths['test_stim_path'],eeg=True)
#add questionairre to df

##start EEG
# thread_q = Queue()
#
# #create and start thread
# x = threading.Thread(target=load_data,args=(subject_info,paths,eeg_df,thread_q,))
# x.start()

experiment_run(subject_info,win,cwd,fixation,paths,rest,paths['test_stim_path'],timing,df,break_stim,end_stim,test=True)

#stop thread
# process_q.put("Stop")
# x.join()

#close window and quit core
win.close()
core.quit()
