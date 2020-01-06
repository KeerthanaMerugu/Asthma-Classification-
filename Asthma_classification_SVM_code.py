####importing packages#######
import os
import scipy.io
import glob
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn import preprocessing
from keras.utils import to_categorical 
from sklearn.metrics import recall_score,confusion_matrix,accuracy_score,f1_score,classification_report
from sklearn.model_selection import GridSearchCV,train_test_split,cross_val_score,ParameterGrid

#path1='/home/mpskkeerthu/Documents/open_smile_asthma_healhty/';
##### dataset path#############
path1='/home/sgeadmin/keerthana/open_smile_asthma_healhty/';
if not os.path.exists(os.path.dirname(path1+'model')):
       os.makedirs(os.path.dirname(path1+'model'))
#os.mkdir(path1+'model')
predicted_prob=[];predicted_values=[];
recall_score_values=[];
accuracy_score_array=[];accuracy_test_array=[];accuracy_val_array=[];accuracy_train_array=[];
from_cm_accuracy_array=[];
accuracy_avg_array=[];
t_p_array=[];t_n_array=[];f_p_array=[];f_n_array=[];parameters=[];

def svc_param_selection(X,Y,x_t,y_t,x,y,s,num1):
    ###########path to save the trained models########
    path_f='/home/sgeadmin/keerthana/open_smile_asthma_healhty/model/';
    print("training the model and finding parameters:");
    ########tuning the SVM parameters##############
    Cs = 2.**np.arange(-5,2)
    param_grid = {'C': Cs};
    ######### building the SVM model###############
    model=SVC(kernel='linear',probability=True)
    grid_search = GridSearchCV(model,param_grid,cv=5)
    ############training the model###########
    grid_search.fit(X,Y)
    accuracy_test=grid_search.score(x_t,y_t)
    accuracy_val=grid_search.score(x,y);
    accuracy_train=grid_search.score(X,Y);
    print("avg_accuracy---:",grid_search.grid_scores_); 
    #print("results_grid_Search---:",cv_results_);
    print("accuracy_test-----:",accuracy_test);
    print("accuracy_val----:",accuracy_val);
    print("accuracy_train----",accuracy_train);
    accuracy_test_array.append(accuracy_test);
    accuracy_val_array.append(accuracy_val);
    accuracy_train_array.append(accuracy_train)
    accuracy_avg_array.append(grid_search.grid_scores_);### gridsearch scores
    ###prediction
    y_pred=grid_search.predict(x_t);
    predicted_values.append(y_pred);
    
    # predict probabilities
    y_prob = grid_search.predict_proba(x_t)
    accuracy_scores=accuracy_score(y_t,y_pred,normalize=True);
    print('accuracy_scores=',accuracy_scores);
    accuracy_score_array.append(accuracy_scores)
    #predicted_prob.append(y_prob);
    recall_score_values.append(recall_score(y_t,y_pred)) 
    
    ###saving the model
    os.chdir(path_f);
    pkl_filename='model_'+s+str(num1)+'.pkl';
    with open(pkl_filename, 'wb') as file:
        pickle.dump(grid_search, file)

    cm=confusion_matrix(y_t,y_pred);
    print("-------",s,num1,"----------")
    print("length of y_train===",len(Y))
    print("class_1:",np.sum(Y==1))
    print("class_0:",np.sum(Y==0))
    print("length of y_test===",len(y_t))
    print("class_1:",np.sum(y_t==1))
    print("class_0:",np.sum(y_t==0))
    t_n=cm[0][0];
    f_p=cm[0][1];
    f_n=cm[1][0];
    t_p=cm[1][1];
    total=float((cm[0][0]+cm[1][1]))/float((cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]));
    print("conf_matrix:",cm);
    t_p_array.append(t_p);
    t_n_array.append(t_n);
    f_p_array.append(f_p);
    f_n_array.append(f_n);
    print("true_p_rate----:",t_p);
    print("false_p_rate----:",f_p);
    print("true_n_rate----:",t_n);
    print("false_n_rate----:",f_n);
    print("from_cm_accuracy---:",total);
    from_cm_accuracy_array.append(total);
    print(classification_report(y_t,y_pred))
    print(grid_search.best_estimator_.C)
    print("parameters--:",grid_search.best_params_)
    parameters.append(grid_search.best_params_);
    #print("probabilities:",y_prob)


###### phonems which we are used to classify the control and patient######## (each audio file includes these phonems,each phonemis repeated on an average of 5 times  )
# five fold cross validation
# i have created five train.mat files and five test.mat files (each mat file consists of opensmile features(with the dimension ---number of audio files x 6373) of input data)
sounds=['Cough','Inhale','Exhale','Wheeze','Aaa','Eee','Ooo','Uuu','Sss','Zzz','Yee'];
#sounds=['Wheeze','Aaa','Eee'];
for j in range(len(sounds)):
    path=path1+sounds[j]+'_mat';
    for i in range(5):
        os.chdir(path);
        data=(scipy.io.loadmat('train_norm_'+str(i+1)+'.mat'))
        X_train=data['feature1']
        Y_train=data['label']
        X_train=np.nan_to_num(X_train)
       # X_train1 = preprocessing.normalize(X_train)
        y_train=np.ravel(Y_train)
        #os.chdir(path2)
        data=(scipy.io.loadmat('test_norm_'+str(i+1)+'.mat'))
        X_test=data['feature1']
        Y_test=data['label']
        X_test=np.nan_to_num(X_test)
        #X_test1 = preprocessing.normalize(X_test)
        y_test=np.ravel(Y_test)
        X_train1,X_val,y_train1,y_val=train_test_split(X_train,y_train,test_size=0.2)
        svc_param_selection(X_train,y_train,X_test,y_test,X_val,y_val,sounds[j],i)

print("acuuracy_score_array----:",accuracy_score_array.reshape(len(sounds),5));

print("true_p_rate--:",(np.array(t_p_array)).reshape(len(sounds),5));
print("true_n_rate--:",(np.array(t_n_array)).reshape(len(sounds),5));
print("false_p_rate--:",(np.array(f_p_array)).reshape(len(sounds),5));
print("false_n_rate--:",(np.array(f_n_array)).reshape(len(sounds),5));
print("acuuracy_test_array----:",(np.array(accuracy_test_array)).reshape(len(sounds),5));
print("accuracy_val_array----:",(np.array(accuracy_val_array)).reshape(len(sounds),5));
print("from_conf_matrix_accuracy-----:",from_cm_accuracy_array);
#print("accuracy_avg_array---:",accuracy_avg_array.reshape(len(sounds),5));

