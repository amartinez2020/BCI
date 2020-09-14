from analysis_helpers import *

#experiment title
title = "BCI"

#navigate to subject data
subject_info = get_subject_info(title)
os.chdir(f'../data/{subject_info}')

#set classes and classifiers
classes = ['hand','word']
classifiers = {'LDA': LinearDiscriminantAnalysis(),'LR':LogisticRegression(),'SVM':SVC()}
transforms_dict= {'CSP': CSP(n_components=n_components, reg=None, log=True, norm_trace=False)}
transforms = [('CSP',transforms['CSP'])]


#train linear model on given classes and subject
train(subject_info,classes,classifiers,transforms,model_name='LR')
