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
subject_info = get_subject_info(title)

#paths
paths = {'stim_path':os.path.join(cwd,'stim'),'data_path':os.path.join(cwd,'data/'+str(subject_info))}

#setup subject directory
try:
    os.mkdir(paths['data_path'])
except:
    print("====Subject Directory Exists====")

#run questionairre
#pre_questionnaire(subject_info, paths=paths, save=True)

#logging
logging.setDefaultClock(core.Clock())
logging.LogFile(f=(paths['data_path']+"/"+subject_info+"_log.log"),filemode='w',level=logging.DEBUG)
logging.log(level=logging.WARN, msg='warning')
logging.log(level=logging.EXP , msg='experiment')
logging.log(level=logging.DATA, msg='data')
logging.log(level=logging.INFO, msg='info')
logging.info("eeg time: " + str(local_clock()))

#params
# params = {'runs':8,'trials_per_run':30}
#for testing ends of runs
params = {'runs':1,'trials_per_run':6}

#window and frame rate variables
win = visual.Window([1680,1050], fullscr = False, monitor = 'testMonitor', units='deg', color = 'black')
win.logOnFlip(level=logging.EXP, msg='frame flip')
rate = win.getActualFrameRate()

#timing
timing = {'fix':3, 'stim':4.5, 'trial':10}

#stimuli
stim_list = {"word":'Letter.png',"sub":'Subtraction.png', "nav": 'Nav.jpg',"hand":'hand_squeeze.png', "rotate":'Rotation.png',"feet":'Feet.jpg'}

#fixation cross
fixation = fix_stim(win)

#rest stimulus
rest = visual.TextStim(win,text='REST',height=2)

#break and end stims
break_stim = visual.TextStim(win,text='End of run. Press Spacebar to continue.', name='break_stimulus', height=1)
end_stim = visual.TextStim(win,text='End of Experiment! Press Spacebar to exit.', name='end_stimulus',height=1)

#initialize dfs
df = init_df(subject_info,paths,params,stim_list)
eeg_df = init_df(subject_info,paths,params,stim_list,eeg=True)


##start EEG
# thread_q = Queue()
# #set_executable(os.path.join(sys.exec_prefix,'pythonw.exe'))
#
# #create and start thread
# x = threading.Thread(target=load_data,args=(subject_info,paths,eeg_df,thread_q,))
# x.start()

#start trials
experiment_run(subject_info,win,cwd,fixation,paths,rest,stim_list,timing,df,break_stim,end_stim)

# #stop thread
# process_q.put("Stop")
# x.join()

#close window and quit core
win.close()
core.quit()
