#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 20:57:30 2017

@author: khchanaq
"""

#Basic Packages
import numpy as np
import pandas as pd
#Import ML Models that used
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.ensemble import GradientBoostingRegressor as gbr
from sklearn.ensemble import ExtraTreesRegressor as etr
from sklearn.svm import SVR as svr
from sklearn.linear_model import LinearRegression as lr
from xgboost import XGBRegressor as xgb
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import Dropout
from keras.layers import Input
from keras.models import Model
#Import Imputer for missing value fill-in
from sklearn.preprocessing import Imputer
#StandardScaler for Feature Scaling
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
#Feature Reduction with PCA package
from sklearn.decomposition import PCA
# Cross-validation with GridSearch
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import KFold
from sklearn.model_selection import cross_val_score
#from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute
from imputer import Imputer as KnnIm
import time
import datetime

###############################--------Global Environment Variable--------###############################
Env_var = {'IsRawData' : 0,
           'WaytoFillNan' : 'knn', 
           'FeatureScaling' : 0, 
           'Pca' : 0,
           'Model' : 'xgb',
           'GridSearch' : 0,
           'DataTrim' : 0
           }


###############################--------Data-preprocessing Function--------###############################
def convert_to_float(dataset):

    for i in range(0,len(dataset[1])):
        for j in range(0, len(dataset)):
            if type(dataset[j,i]) is str:
                if dataset[j,i] == '#!':
                    dataset[j,i] = np.nan
                else:
                    dataset[j,i] = float(dataset[j,i].replace(',','').replace("-", ""))

    return dataset

def fillMissingValue(df, fy, WaytoFillNan = Env_var.get('WaytoFillNan')):
    train_data_temp = df[df.iloc[:,fy].notnull()]  
    test_data_temp = df[df.iloc[:,fy].isnull()]  
    train_y=train_data_temp.iloc[:,fy]
    train_X=train_data_temp.copy()
    train_X = train_X.drop(train_X.columns[fy], axis = 1)
    test_X = test_data_temp.copy()
    test_X = test_X.drop(test_X.columns[fy], axis =1)
    mixed_X = Imputer().fit_transform(train_X.append(test_X, ignore_index=True))
    length_train = len(train_X)
    train_X = mixed_X[:length_train,:]
    test_X = mixed_X[length_train:,:]
    
    if (WaytoFillNan == 'rfr'):
        print ("Try to fill-up value with rfr")
        rfr_regressor=rfr(n_estimators=100, verbose = 5, n_jobs = -1)
        rfr_regressor.fit(train_X,train_y)
        y_pred = rfr_regressor.predict(test_X)
        print (y_pred)
        df.iloc[:,fy] = df.iloc[:,fy].fillna(value = pd.Series(data = y_pred))

    elif (WaytoFillNan == 'mean'):
        print ("Try to fill-up value with mean")
        df.iloc[:,fy] = df.iloc[:,fy].fillna(value = np.mean(df.iloc[:,fy]))
        
    elif (WaytoFillNan == 'ffill'):
        df.iloc[:,fy] = df.iloc[:,fy].fillna(method = 'ffill')

    elif (WaytoFillNan == 'bfill'):
        df.iloc[:,fy] = df.iloc[:,fy].fillna(method = 'bfill')

    elif (WaytoFillNan == 'knn'):
        impute = KnnIm()
        df.iloc[:,fy] = (impute.knn(X=df, column=fy, k=10))[:,fy]
    
    return df

def findMissingValue(X):
    #Check out Empty Data
    EmptyDataList = []
    for i in range(0, len(X.iloc[0])):
        if(np.any(np.isnan(X.iloc[:,i]))):
            EmptyDataList.append(i)

    return EmptyDataList


###############################--------Metrics Setup--------###############################
def rmsle(predicted, actual):
    
    return np.sqrt(np.nansum(np.square(np.log(predicted + 1) - np.log(actual + 1)))/float(len(predicted)))

def rmse(predicted, actual):
    
    return np.sqrt(np.nansum(np.square(predicted - actual))/float(len(predicted)))


def AutoGridSearch(parameters, regressor, train_data_X, train_data_y):

    scorer = make_scorer(rmsle, greater_is_better=False)
#    while True:
    
    #Perform grid search on the classifier using 'scorer' as the scoring method
    grid_obj = GridSearchCV(estimator = regressor,
                               param_grid = parameters,
                               scoring = scorer,
                               cv = 5,
                               verbose=10,
                               n_jobs = -1)
    
    #Fit the grid search object to the training data and find the optimal parameters
    grid_fit = grid_obj.fit(train_data_X, train_data_y)
    
    #if(best_score < grid_fit.best_score_):
    best_score = grid_fit.best_score_

    best_parameters = grid_fit.best_params_
    
    best_estimator = grid_fit.best_estimator_
    
    print("Best: %f using %s" % (grid_fit.best_score_, grid_fit.best_params_))
    for params, mean_score, scores in grid_fit.grid_scores_:
        print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))
    
    return best_score, best_parameters, best_estimator

###############################--------Model Setup--------###############################
def ann_model(dropout_rate=0.0, length = 559):
    # create model
    model = Sequential()
    model.add(Dense(1120, input_dim=length, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(599, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(dropout_rate))    
    model.add(Dense(299, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(145, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(74, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(37, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(14, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(7, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(4, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(2, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_logarithmic_error', optimizer='RMSprop')
    return model

class Ensemble(object):
    def __init__(self, n_folds, stacker, base_models):
        self.n_folds = n_folds
        self.stacker = stacker
        self.base_models = base_models
    
    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)
        
        folds = list(KFold(len(y), n_folds=self.n_folds, random_state=2017))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))

        for i, clf in enumerate(self.base_models):
            S_test_i = np.zeros((T.shape[0], len(folds)))
            print ("we are now in iteration : " + str(i))

            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                # y_holdout = y[test_idx]
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_holdout)[:]
                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict(T)[:]

            S_test[:, i] = S_test_i.mean(1)

        self.stacker.fit(S_train, y)
        y_pred = self.stacker.predict(S_test)[:]
        return y_pred
        
###############################--------Submission Function--------###############################
def submit(test_data, y_pred, filename, training, scores = 0):

    results = pd.DataFrame({
    'id' : test_data['id'].astype(np.int32),
    'price_doc' : y_pred
    })

    scoreStr = "{:.3f}".format(scores)
    
    if(training):
        results.to_csv("./internalsubmission/" + Env_var.get('Model') + "/" + filename + "_" + scoreStr + ".csv", index=False)
    else:
        results.to_csv("./submission/" + Env_var.get('Model') + "/" + filename + "_" + scoreStr + ".csv", index=False)

###############################--------Result Visualization--------###############################
def evaluate(learner, sample_size, X_train, y_train, X_test, y_test): 
    
    results = {}
    
    start = time() # Get start time
    learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time() # Get end time
    
    results['train_time'] = end - start
        
    #       then get predictions on the first 300 training samples
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train)
    end = time() # Get end time
    
    results['pred_time'] = end - start
            
    results['rmsle_train'] = rmsle(y_train, predictions_train)

    results['rmsle_test'] = rmsle(y_test, predictions_test)
           
    # Success
    print ("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))
        
    # Return the results
    return results

def visualize(results):
    import matplotlib.pyplot as pl
    import matplotlib.patches as mpatches
    fig, ax = pl.subplots(2, 3, figsize = (11,7))
    bar_width = 0.3
    colors = ['#A00000','#00A0A0','#00A000']

    for k, learner in enumerate(results.keys()):
        for j, metric in enumerate(['train_time', 'rmsle_train', 'pred_time', 'rmsle_train']):
            for i in np.arange(3):
                
                # Creative plot code
                ax[j/3, j%3].bar(i+k*bar_width, results[learner][i][metric], width = bar_width, color = colors[k])
                ax[j/3, j%3].set_xticks([0.45, 1.45, 2.45])
                ax[j/3, j%3].set_xticklabels(["1%", "10%", "100%"])
                ax[j/3, j%3].set_xlabel("Training Set Size")
                ax[j/3, j%3].set_xlim((-0.1, 3.0))
    
    # Add unique y-labels
    ax[0, 0].set_ylabel("Time (in seconds)")
    ax[0, 1].set_ylabel("Accuracy Score")
    ax[1, 0].set_ylabel("Time (in seconds)")
    ax[1, 1].set_ylabel("Accuracy Score")
    
    # Add titles
    ax[0, 0].set_title("Model Training")
    ax[0, 1].set_title("Accuracy Score on Training Subset")
    ax[1, 0].set_title("Model Predicting")
    ax[1, 1].set_title("Accuracy Score on Testing Set")
    
    # Set y-limits for score panels
    ax[0, 1].set_ylim((0, 1))
    ax[0, 2].set_ylim((0, 1))
    ax[1, 1].set_ylim((0, 1))
    ax[1, 2].set_ylim((0, 1))

    # Create patches for the legend
    patches = []
    for i, learner in enumerate(results.keys()):
        patches.append(mpatches.Patch(color = colors[i], label = learner))
    pl.legend(handles = patches, bbox_to_anchor = (-.80, 2.53), \
               loc = 'upper center', borderaxespad = 0., ncol = 3, fontsize = 'x-large')
    
    # Aesthetics
    pl.suptitle("Performance Metrics for Three Supervised Learning Models", fontsize = 16, y = 1.10)
    pl.tight_layout()
    pl.show()

def main_func(Env_var):
    
    best_parameters = {}
    score = 0
 

###############################--------Data Pre-processing--------###############################
    if (Env_var.get('IsRawData') == 0):
    
        #Read dataset with pandas
        if (Env_var.get('WaytoFillNan') == 'rfr'):
            train_data_X = pd.read_csv("./training/train_data_X_RFECV.csv", quoting = 2)
            test_data_X = pd.read_csv("./training/test_data_X_RFECV.csv", quoting = 2)
        elif (Env_var.get('WaytoFillNan') == 'mean'):
            train_data_X = pd.read_csv("./training/train_data_X_mean.csv", quoting = 2)
            test_data_X = pd.read_csv("./training/test_data_X_mean.csv", quoting = 2)
        elif (Env_var.get('WaytoFillNan') == 'ffill'):
            train_data_X = pd.read_csv("./training/train_data_X_ffill.csv", quoting = 2)
            test_data_X = pd.read_csv("./training/test_data_X_ffill.csv", quoting = 2)
        elif (Env_var.get('WaytoFillNan') == 'bfill'):
            train_data_X = pd.read_csv("./training/train_data_X_bfill.csv", quoting = 2)
            test_data_X = pd.read_csv("./training/test_data_X_bfill.csv", quoting = 2)
        elif (Env_var.get('WaytoFillNan') == 'knn'):
            train_data_X = pd.read_csv("./training/train_data_X_knn_cleaned.csv", quoting = 2)
            test_data_X = pd.read_csv("./training/test_data_X_knn_cleaned.csv", quoting = 2)    
    
        train_data = pd.read_csv("train.csv", quoting = 2)
        train_data_y = train_data.iloc[:,-1]
        test_data = pd.read_csv("test.csv", quoting = 2)
        '''
        train_lat_lon = pd.read_csv("./externaldata/train_lat_lon.csv", quoting = 2)
        test_lat_lon = pd.read_csv("./externaldata/test_lat_lon.csv", quoting = 2)
        train_lat_lon = train_lat_lon.iloc[:,2:4]
        test_lat_lon = test_lat_lon.iloc[:,2:4]
        train_data_X = pd.concat([train_data_X, train_lat_lon], axis = 1, ignore_index = True)
        test_data_X = pd.concat([test_data_X, test_lat_lon], axis = 1, ignore_index = True)
        '''
        
        print (np.any(np.isnan(train_data_X)))
        print (np.any(np.isnan(test_data_X)))
        print (len(train_data_X))
        print (len(test_data_X))
        print (len(train_data_X.iloc[0]))
        print (len(test_data_X.iloc[0]))
        
    
    else:
        
        train_data = pd.read_csv("train.csv", quoting = 2)
        test_data = pd.read_csv("test.csv", quoting = 2)
        macro_data = pd.read_csv("./training/macro_external_clean_knn.csv", quoting = 2)
        macro_data_external = pd.read_csv("macro_external_data.csv", quoting = 2)
        train_lat_lon = pd.read_csv("./externaldata/train_lat_lon.csv", quoting = 2)
        test_lat_lon = pd.read_csv("./externaldata/test_lat_lon.csv", quoting = 2)
        train_lat_lon = train_lat_lon.iloc[:,2:4]
        test_lat_lon = test_lat_lon.iloc[:,2:4]
        
        #Separate data into X,y
        train_data_y = train_data.iloc[:,-1]
        train_data_X = train_data.iloc[:,1:-1]
        test_data_X = test_data.iloc[:,1:]
        #join macro environment
        train_data_X = pd.merge(train_data_X, macro_data, on='timestamp');
        train_data_X = train_data_X.iloc[:,1:]
        #join macro environment
        test_data_X = pd.merge(test_data_X, macro_data, on='timestamp');
        test_data_X = test_data_X.iloc[:,1:]
        
        train_data_X = pd.concat([train_data_X, train_lat_lon], axis = 1, ignore_index = True)
        test_data_X = pd.concat([test_data_X, test_lat_lon], axis = 1, ignore_index = True)
        
        #One hot encoder for categorial variable
        mixed_data_X = train_data_X.append(test_data_X, ignore_index=True)
        length_train = len(train_data_X)
        mixed_data_X = pd.get_dummies(mixed_data_X)


        new_macro_2 = pd.get_dummies(new_macro_2)
        time.mktime(t.timetuple())
        timestamp_d = np.zeros(len(new_macro))
        offset = datetime.datetime.strptime(mixed_data_X.iloc[0,0], "%Y-%m-%d")
        for i in range(0, len(new_macro)):
            temp = new_macro.iloc[i, 0]  - offset
            timestamp_d[i] = temp.days
        
        date_list_pd = pd.DataFrame(date_list)
        new_list = np.zeros(len(date_list_pd))
        for i in range(0, len(date_list)):
            new_list = date_list_pd.iloc[i,0].strftime("%Y-%m-%d")
        
        new_macro = pd.merge(dates, macro_data_external, how='left', on='date')
        new_macro = new_macro.fillna(method = 'ffill')
        
        macro_data_external['date'] = pd.to_datetime(macro_data_external['date'])
        
        dates = pd.DataFrame(dates)           
        macro_data_external = macro_data_external.reindex(dates, method='ffill')
        
        new_macro = pd.concat([macro_data, new_macro], axis = 1, ignore_index = True)
        macro_data = macro_data.reset_index()
        macro_data = macro_data.iloc[:,1:]
        timestamp_d = pd.DataFrame(timestamp_d)
        new_macro = new_macro.iloc[:,100:]
        new_macro_2 = new_macro.iloc[:2394,1:]
        new_macro_2 = pd.concat([macro_data, new_macro_2], axis = 1, ignore_index = True)

        pd.DataFrame(new_macro_2).to_csv("./training/macro_external_clean.csv", index=False)

        emptyList_mix = findMissingValue(new_macro_2)
  
        for i in emptyList_mix:
            print ("Filling-up column : " + str(i))
            new_macro_2 = fillMissingValue(new_macro_2, i)
            pd.DataFrame(new_macro_2).to_csv("./training/macro_external_clean_knn.csv", index=False)
            print ("Wrote to hdd til column : " + str(i))
           
        emptyList_mix = findMissingValue(mixed_data_X)
        
     

        for i in emptyList_mix:
            print ("Filling-up column : " + str(i))
            mixed_data_X = fillMissingValue(mixed_data_X, i)
            if(np.isnan(np.min(mixed_data_X.iloc[:,i])) == False):
                train_data_X = mixed_data_X.iloc[:length_train,:]
                test_data_X = mixed_data_X.iloc[length_train:,:]
                if (Env_var.get('WaytoFillNan') == 'rfr'):
                    pd.DataFrame(train_data_X).to_csv("./training/train_data_X_rfr100.csv", index=False)
                    pd.DataFrame(test_data_X).to_csv("./training/test_data_X_rfr100.csv", index=False)
                elif (Env_var.get('WaytoFillNan') == 'mean'):
                    pd.DataFrame(train_data_X).to_csv("./training/train_data_X_mean.csv", index=False)
                    pd.DataFrame(test_data_X).to_csv("./training/test_data_X_mean.csv", index=False)
                elif (Env_var.get('WaytoFillNan') == 'ffill'):
                    pd.DataFrame(train_data_X).to_csv("./training/train_data_X_ffill.csv", index=False)
                    pd.DataFrame(test_data_X).to_csv("./training/test_data_X_ffill.csv", index=False)
                elif (Env_var.get('WaytoFillNan') == 'bfill'):
                    pd.DataFrame(train_data_X).to_csv("./training/train_data_X_bfill.csv", index=False)
                    pd.DataFrame(test_data_X).to_csv("./training/test_data_X_bfill.csv", index=False)
                elif (Env_var.get('WaytoFillNan') == 'knn'):
                    pd.DataFrame(train_data_X).to_csv("./training/train_data_X_div_30.csv", index=False)
                    pd.DataFrame(test_data_X).to_csv("./training/test_data_X_div_30.csv", index=False)
   
                print ("Wrote to hdd til column : " + str(i))
    
        print (np.any(np.isnan(mixed_data_X)))
        print (np.any(np.isnan(train_data_X)))
        print (np.any(np.isnan(test_data_X)))
        print (len(train_data_X))
        print (len(test_data_X))
        print (len(train_data_X.iloc[0]))
        print (len(test_data_X.iloc[0]))

date_list = [start_date + datetime.timedelta(days=x) for x in range(0, numdays)]

start_date =  pd.Timestamp(macro_data_external.iloc[:,0].min()) - pd.DateOffset(day=1)
end_date =  pd.Timestamp(macro_data_external.iloc[:,0].max()) + pd.DateOffset(day=31)

dates = pd.date_range(start_date, end_date, freq='D')
dates.name = 'date'



numdays = (end_date - start_date).days  


    for i in range(0, len(mixed_data_X.iloc[0])):
        #mixed_data_X.iloc[:,i].describe()
        print (str(i) + "turn" )
        i = 9
        Mean = np.mean(mixed_data_X.iloc[:,i])
        mixed_data_X.iloc[:,i].describe()
        if (Mean > 76):
            mintemp = mixed_data_X[mixed_data_X.iloc[:,i] > 4]
            mintemp = mixed_data_X[mixed_data_X.iloc[:,i] == 0]
            mixed_data_X.iloc[mintemp.index,i] = np.nan
            print (str(i) + "has 0 value" )
        pd.DataFrame(train_data_X).to_csv("./training/train_data_X_c_t_n.csv", index=False)
        pd.DataFrame(test_data_X).to_csv("./training/test_data_X__c_t_n.csv", index=False)
       
    
    if (Env_var.get('DataTrim') == 1):
        train_data_X = train_data_X.iloc[3000:,:]
        train_data_y = train_data_y.iloc[3000:]
        macro_data = macro_data.iloc[90:]
        macro_data_external = macro_data_external.iloc[:84]
    
    if (Env_var.get('FeatureScaling') == 1):
        sc_X = MinMaxScaler()
        train_data_X = sc_X.fit_transform(train_data_X)
        test_data_X = sc_X.transform(test_data_X)
        train_data_y = np.log(1+train_data_y)

    if (Env_var.get('Pca') == 1):
        pca = PCA(n_components = 300).fit(train_data_X)    
        train_data_X = pca.transform(train_data_X)
        test_data_X = pca.transform(test_data_X)
    
    ###############################--------Model Setup--------###############################
    ann_regressor = KerasRegressor(build_fn=ann_model, epochs=30, batch_size=10, verbose=1)
    
    xgb_regressor = xgb(learning_rate = 0.0825, min_child_weight = 1, max_depth = 7, subsample = 0.8, verbose = 10, random_state = 2017, n_jobs = -1, eval_metric = "rmse")
    
    rfr_regressor = rfr(max_features = 0.9, min_samples_leaf = 50)
    
    gbr_regressor = gbr(n_estimators = 200, verbose = 5, learning_rate = 0.08, max_depth = 7, max_features = 0.5, min_samples_leaf = 50, subsample = 0.8, random_state = 2017)
    
    etr_regressor = etr(n_estimators = 200, verbose = 10, max_depth = 7, min_samples_leaf = 100, max_features = 0.9, min_impurity_split = 100, random_state = 2017)
    
    lr_regressor = lr()
    
    svr_regressor = svr(verbose = 10)
    
    ensemble = Ensemble(n_folds = 5,stacker =  lr_regressor,base_models = [ann_regressor, xgb_regressor, rfr_regressor, gbr_regressor, etr_regressor])
    
    
    ###############################--------Grid Search--------###############################

  
    if (Env_var.get('GridSearch') == 1):
        
        if (Env_var.get('Model') == 'ann'):
            dropout_rate = [0.0, 0.001, 0.01]
            ann_parameters = dict(dropout_rate=dropout_rate)
            
            score, best_parameters, best_model = AutoGridSearch(ann_parameters,ann_regressor, train_data_X, train_data_y)
    
        elif (Env_var.get('Model') == 'xgb'):
            xgb_parameters = {'learning_rate' : [0.01, 0.1, 1.0], 
                     'min_child_weight' : [1, 3, 5],
                     'max_depth' : [3, 5 ,7],
                     'subsample' : [0.6, 0.8, 1.0]}
    
            score, best_parameters, best_model = AutoGridSearch(xgb_parameters,xgb_regressor, train_data_X, train_data_y)
    
        elif (Env_var.get('Model') == 'gbr'):
            #Create the parameters list you wish to tune
            gbr_parameters = {'learning_rate': [0.01, 0.1, 0.5],
                              'max_depth': [6, 7, 8],
                          'min_samples_leaf': [25, 50, 75],
                          'subsample': [0.8],
                          'max_features': [0.3, 0.5, 0.7],
                          'random_state': [2017]}
            
            score, best_parameters, best_model = AutoGridSearch(gbr_parameters,gbr_regressor, train_data_X, train_data_y)
    
        elif (Env_var.get('Model') == 'etr'):
            #Create the parameters list you wish to tune
            etr_parameters = {'max_depth': [3, 5, 7],
                  'min_samples_leaf': [50, 100, 150],
                  'max_features': [0.1, 0.5, 0.9],
                  'min_impurity_split': [50, 100, 150],
                  'random_state': [2017]}
    
            score, best_parameters, best_model = AutoGridSearch(etr_parameters,etr_regressor, train_data_X, train_data_y)
            
        elif (Env_var.get('Model') == 'rfr'):
            #Create the parameters list you wish to tune
            rfr_parameters = {'max_features': [0.7, 0.8, 0.9],
                              'min_samples_leaf': [25, 50, 75],
                              'random_state': [2017]}
            
            
            score, best_parameters, best_model = AutoGridSearch(rfr_parameters,rfr_regressor, train_data_X, train_data_y)
    
            
        elif (Env_var.get('Model') == 'svr'):
            svr_parameters = {'C': [0.001, 0.1, 1, 100],
                      'degree': [1, 3, 5],
                      'gamma': [0.01, 0.1, 1]
                      }
    
            score, best_parameters, best_model = AutoGridSearch(svr_parameters,svr_regressor, train_data_X, train_data_y)
    
        if (Env_var.get('Model') == 'ensemble'):
            CVscore = []
            folds = list(KFold(len(train_data_y), n_folds= 5, random_state=2017))
            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = train_data_X[train_idx]
                y_train = train_data_y[train_idx]
                X_holdout = train_data_X[test_idx]
                y_holdout = train_data_y[test_idx]
                y_pred = ensemble.fit_predict(X_train, y_train, X_holdout)
                score = rmsle(y_pred, y_holdout)
                CVscore.append(score)
    
            score = np.mean(CVscore)
    
    else:
        
        best_model = xgb_regressor
        
        if (Env_var.get('Model') == 'ann'):
            best_model = ann_regressor
        elif (Env_var.get('Model') == 'xgb'):
            best_model = xgb_regressor
            print ("Test for xgb run")
        elif (Env_var.get('Model') == 'gbr'):
            best_model = gbr_regressor
        elif (Env_var.get('Model') == 'etr'):
            best_model = etr_regressor
        elif (Env_var.get('Model') == 'rfr'):
            best_model = rfr_regressor
        elif (Env_var.get('Model') == 'svr'):
            best_model = svr_regressor
        
        scorer = make_scorer(rmse, greater_is_better=False)
        fold = KFold(len(train_data_y), n_folds=5, random_state=2017)
        
        if (Env_var.get('Model') == 'ensemble'):
            CVscore = []
            folds = list(KFold(len(train_data_y), n_folds= 5, random_state=2017))
            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = train_data_X[train_idx]
                y_train = train_data_y[train_idx]
                X_holdout = train_data_X[test_idx]
                y_holdout = train_data_y[test_idx]
                y_pred = ensemble.fit_predict(X_train, y_train, X_holdout)
                temp = rmsle(y_pred, y_holdout)
                CVscore.append(temp)
            
            score = np.mean(CVscore)
        else:
            score = cross_val_score(best_model, train_data_X, train_data_y, cv = 5, verbose = 10, scoring = scorer)
            score = np.mean(score)
    

    
    ###############################--------Output--------###############################
    from time import gmtime, strftime
    curTime = strftime("%Y-%m-%d %H:%M:%S", gmtime())    
    filename = Env_var.get('Model') +'_' + curTime + '_' + Env_var.get('WaytoFillNan')
    
    if (Env_var.get('Model') == 'ensemble'):
            
        y_pred_train = ensemble.fit_predict(train_data_X, train_data_y, train_data_X)
        y_pred_test = ensemble.fit_predict(train_data_X, train_data_y, test_data_X)
    
    else:
        best_model.fit(train_data_X, train_data_y)
#        y_pred_train = best_model.predict(train_data_X)
#        y_pred_train = np.exp(y_pred_train) - 1
        y_pred_test = best_model.predict(test_data_X)
        y_pred_test = np.exp(y_pred_test) - 1

    submit(test_data, y_pred_test, filename, training = False, scores = score)
#    submit(train_data, y_pred_train, filename, training = True, scores = score)

    importance = best_model.feature_importances_
    feature_order = np.argsort(importance)[::-1]


    for i in feature_order[:30]:
        for j in feature_order[:30]:
            if (i != j) and (i < j):
                print ("Generating Feature : feature " + str(i) + " / " + "feature " + str(j))
                temp = mixed_data_X.iloc[:,i] / mixed_data_X.iloc[:,j] 
                mixed_data_X = pd.concat([mixed_data_X, temp], axis = 1, ignore_index = True)
       

    return best_parameters, score


###########################################################################
loop_result = []

Env_var = {'IsRawData' : 0,
           'WaytoFillNan' : 'knn', 
           'FeatureScaling' : 1, 
           'Pca' : 0,
           'Model' : 'svr',
           'GridSearch' : 1,
           'DataTrim' : 0
           }

loop_result.append([Env_var, main_func(Env_var)])

Env_var = {'IsRawData' : 0,
           'WaytoFillNan' : 'knn', 
           'FeatureScaling' : 1, 
           'Pca' : 0,
           'Model' : 'ensemble',
           'GridSearch' : 0,
           'DataTrim' : 0
           }

loop_result.append([Env_var, main_func(Env_var)])

Env_var = {'IsRawData' : 0,
           'WaytoFillNan' : 'knn', 
           'FeatureScaling' : 1, 
           'Pca' : 0,
           'Model' : 'ann',
           'GridSearch' : 1,
           'DataTrim' : 0
           }


loop_result.append([Env_var, main_func(Env_var)])


train_data_y = np.exp(train_data_y) - 1

train_data_y.describe()

train_pred_data = pd.read_csv("./internalsubmission/ensemble/0.471_submission.csv", quoting = 2)

train_pred_data = train_pred_data.iloc[:,1]

y_diff = train_data_y

y_diff.describe()

outlier_greater = np.where(y_diff > np.mean(y_diff) + 2 * np.std(y_diff))

outlier_less =  np.where(y_diff < 1000000)

y_diff_outlier = y_diff[outlier_less[0]]

train_data_X = pd.DataFrame(train_data_X)

train_data_X = train_data_X.drop(train_data_X.index[outlier_less[0]])

train_data_y = train_data_y.drop(train_data_y.index[outlier_less[0]])


anomaly_greater = outlier_greater[0]

anomaly_less = outlier_less[0]

anomaly_y = np.ones(len(y_diff))

anomaly_y[anomaly_greater] = 2

anomaly_y[anomaly_less] = 0

from xgboost import XGBClassifier as xgbc

xgb_classifier = xgbc(learning_rate = 0.0825, min_child_weight = 1, max_depth = 7, subsample = 0.8, verbose = 10, random_state = 2017)

xgb_classifier.fit(train_data_X, anomaly_y, eval_metric = 'merror')

y_pred = xgb_classifier.predict(train_data_X)

y_pred_test = xgb_classifier.predict(test_data_X)

len(y_pred_test == 1.0)

from sklearn.metrics import accuracy_score

acc_score = accuracy_score(anomaly_y, y_pred)

from sklearn.metrics import confusion_matrix

results_3 = confusion_matrix(anomaly_y, y_pred)

from sklearn.metrics import fbeta_score

fbeta_scorer = make_scorer(fbeta_score, beta = 0.5, average = 'micro')

result_score = cross_val_score(xgb_classifier, train_data_X, anomaly_y, cv = 5, verbose = 10, scoring = acc_score)

anomaly_y = pd.DataFrame(anomaly_y)

train_data_X = pd.concat([train_data_X, anomaly_y], axis = 1, ignore_index = True)
 
y_pred_test = pd.DataFrame(y_pred_test)    

test_data_X = pd.concat([test_data_X, y_pred_test], axis = 1, ignore_index = True)

test_data_X = test_data_X.iloc[:7662]

score_ano = cross_val_score(xgb_classifier, train_data_X, anomaly_y, cv = 5, verbose = 10, scoring = fbeta_scorer)
          
test_data_X.iloc[:,0] = test_data_X.iloc[:,0] ** 2

train_data_X.iloc[:,0] = train_data_X.iloc[:,0] ** 2



import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV

rfecv = RFECV(estimator=xgb_regressor, step=50, cv=2,
              scoring=scorer, verbose = 1)
rfecv.fit(train_data_X, train_data_y)

print("Optimal number of features : %d" % rfecv.n_features_)

ranking = rfecv.ranking_
print (ranking)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(0, len(rfecv.grid_scores_) + 0.1), rfecv.grid_scores_)
plt.show()

train_encoded_X_300, test_encoded_X_300 = autoencoder(train_data_X, test_data_X)

features = np.where(ranking > 1)
len(features[0])

train_data_X = train_data_X.drop(train_data_X.columns[features[0]], axis = 1)

test_data_X = test_data_X.drop(test_data_X.columns[features[0]], axis = 1)



L1_dict = [0, 10e-7, 10e-8]

score_min = 1

for L1 in L1_dict:
    train_temp, test_temp, score_temp = autoencoder(train_data_X, test_data_X, L1 = L1)
    if score_temp < score_min:
        train_encoded_X = train_temp
        test_encoded_X = test_temp
        score_min = score_temp
        best_L1 = L1

from keras import regularizers

######


def autoencoder(X_train, X_test, ncol = 566, L1 = 10e-5):
        
    input_dim = Input(shape = (ncol, ))
    # DEFINE THE DIMENSION OF ENCODER ASSUMED 3
    encoding_dim = 436
    # DEFINE THE ENCODER LAYERS
    encoded1 = Dense(872, activation = 'relu', activity_regularizer=regularizers.l1(L1))(input_dim)
    encoded2 = Dense(736, activation = 'relu', activity_regularizer=regularizers.l1(L1))(encoded1)
    encoded3 = Dense(618, activation = 'relu', activity_regularizer=regularizers.l1(L1))(encoded2)
    encoded4 = Dense(509, activation = 'relu', activity_regularizer=regularizers.l1(L1))(encoded3)
    encoded5 = Dense(480, activation = 'relu', activity_regularizer=regularizers.l1(L1))(encoded4)
    encoded6 = Dense(encoding_dim, activation = 'relu')(encoded5)
    # DEFINE THE DECODER LAYERS
    decoded3 = Dense(480, activation = 'relu', activity_regularizer=regularizers.l1(L1))(encoded6)
    decoded4 = Dense(509, activation = 'relu', activity_regularizer=regularizers.l1(L1))(decoded3)
    decoded5 = Dense(618, activation = 'relu', activity_regularizer=regularizers.l1(L1))(decoded4)
    decoded6 = Dense(736, activation = 'relu', activity_regularizer=regularizers.l1(L1))(decoded5)
    decoded7 = Dense(872, activation = 'relu', activity_regularizer=regularizers.l1(L1))(decoded6)
    decoded8 = Dense(ncol, activation = 'sigmoid')(decoded7)
    # COMBINE ENCODER AND DECODER INTO AN AUTOENCODER MODEL
    autoencoder = Model(input = input_dim, output = decoded8)
    # CONFIGURE AND TRAIN THE AUTOENCODER
    autoencoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')
    autoencoder.fit(X_train, X_train, epochs = 50, batch_size = 100, shuffle = True, validation_data = (X_test, X_test), verbose = 1)
    # THE ENCODER TO EXTRACT THE REDUCED DIMENSION FROM THE ABOVE AUTOENCODER
    encoder = Model(input = input_dim, output = encoded6)
    encoded_input = Input(shape = (encoding_dim, ))
    encoded_out_train = encoder.predict(X_train)
    encoded_out_test = encoder.predict(X_test)
    print (encoded_out_train[0:2])
    print (encoded_out_test[0:2])
    

    return encoded_out_train, encoded_out_test, score

'''
corr_matrix =  (pd.DataFrame(train_data_X).corr())

for i in range (0, len(corr_matrix)):
    for j in range (0, len(corr_matrix)):
        if corr_matrix.iloc[i,j] == 1:
            corr_matrix.iloc[i,j] = 0

BestCorIndex = []
for i in EmptyDataList:
    print (i)
    print (np.max(corr_matrix.iloc[:,i-1]))
    index = np.argmax(corr_matrix.iloc[:,i-1])
    BestCorIndex.append([i,index])

BestCorIndex = pd.DataFrame(BestCorIndex).dropna().values

for i in range(0, 1):
    #Full Data Entry = Training Set
#    for j in range(0, len(train_data_X)):
    train_data_fillempty = train_data_X[:, [BestCorIndex[i][1].astype(np.int32), BestCorIndex[i][0].astype(np.int32)]]

        
    #Empty Data Entry = Testing Set
    test_data_X_fillempty = train_data_X[:, BestCorIndex[i] == np.nan]
    
'''
