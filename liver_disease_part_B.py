
import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.svm import SVC
from scipy.stats import shapiro

csv = pd.read_csv("Indian Liver Patient Dataset (ILPD).csv",header=None)

csv.columns=["Age",'Gender','Tb_Bilirubin','DB_Bilirubin','Alkaline_Phosphotase','sgpt ','Sgot','Protiens','Albumin','Albumin_Globulin','Disease']

#Missing Values
csv.info()
csv.isna().sum()

csv['Albumin_Globulin'] = csv['Albumin_Globulin'].fillna(csv['Albumin_Globulin'].mean())
csv.info()
csv.isna().sum()


#Replace Female with 1 and Male with 0
csv=csv.replace(regex=['Female'],value='1')
csv=csv.replace(regex=['Male'],value='0')



#REPLACE 2->1 AND 1->0 
#0=> nonLiver Patience
#1=> Liver patience
#csv['Disease'].replace(to_replace = 1, value = 0, inplace=True)
#csv['Disease'].replace(to_replace = 2, value = 1, inplace=True)



#Age
resp_age,resp_Tb_Bilirubin=csv.Age,csv.Tb_Bilirubin
shapiro(resp_Tb_Bilirubin)[1]
#Gender

#Create Input && output data 
X_data = csv.drop('Disease',axis=1)
Y_data = csv['Disease']

# Visualize skewed continuous features of original data

X_data.hist(figsize=(14,8))


# Skewed features are Albumin, Direct Bilirubin, A/G ratio, Tota Bilirubin, Total Protein 
#Log-transform the skewed features but also include zero values)
skewed = ['Albumin', 'DB_Bilirubin', 'Tb_Bilirubin', 'Albumin_Globulin', 'Protiens']
X_data[skewed] = X_data[skewed].apply(lambda x: np.log(x + 1))
X_data[skewed] = X_data[skewed]
# Visualize the new log distributions
X_data.hist(figsize=(14,10))

# : One-hot encode the data using pandas.get_dummies()
features = pd.get_dummies(X_data)

encoded = list(features.columns)
print ("{} total features after one-hot encoding.".format(len(encoded)))
print (encoded)

disease=pd.get_dummies(Y_data)
encoded = list(disease.columns)
print ("{} disease columns after one-hot encoding.".format(len(encoded)))
print (disease[1])

#split dataset to train and test data 
X_train,X_test,Y_train,Y_test =train_test_split(X_data,Y_data,test_size=0.25)

def preprocessing(X_train,Y_train,X_test,Y_test,type="MinMax"):

            #NORMALIZATION#
    if type == "Normalization":
        X_train_normalized = tf.keras.utils.normalize(X_train)
        X_test_normalized = tf.keras.utils.normalize(X_test)
        return X_train_normalized,Y_train,X_test_normalized,Y_test

            #NORM-WITH MINMAX#
    elif type == "MinMax":
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_train_minmax = scaler.fit_transform(X_train)
        X_test_minmax = scaler.fit_transform(X_test)
        return X_train_minmax , Y_train ,X_test_minmax ,Y_test

X_train_minmax,Y_train,X_test_minmax,Y_test=preprocessing(X_train,Y_train,X_test,Y_test,type="MinMax")
print(X_train_minmax.shape,X_test_minmax.shape,Y_train.shape,Y_test.shape)

#Function to perform 5 Folds Cross-Validation

kf = KFold(n_splits=5)


def cross_validation(model, _X, _y, _cv=kf):
     #_X array input values
     #_Y out labels
     #cv Determines the number of folds for cross-validation.
     
      _scoring = ['accuracy', 'precision', 'recall', 'f1']

      #Model Training and validation with repeating
      results = cross_validate(estimator=model, X=_X,y=_y,cv=kf,scoring=_scoring,return_train_score=True,verbose=1)
      #Elegxoume tin apododsi pou petuxenei me ta test dedomena pou eginan split apo to kfold
      y_pred = cross_val_predict(model, X_train_minmax,  Y_train, cv=5)
      #pairnoume to accuracy 
      accuracy = accuracy_score(Y_train, y_pred)
      #Geometric mean score
      geom_mean_score=geometric_mean_score(Y_train, y_pred)


      #Fit model without repeating (test prediction with split_testdata)
      model.fit(X_train_minmax,Y_train)
      test_pred = model.predict(X_test_minmax)
      test_accuracy=accuracy_score(Y_test,test_pred)

      print("REPORT CLASSIFICATION FOR train data with repeating 5fold:",classification_report(Y_train,y_pred))
      #info

      #Costum geometric mean calculation:
      sensitivity = recall_score(Y_train , y_pred,average='macro')
      specificity = recall_score(np.logical_not(Y_train) , np.logical_not(y_pred) , average='macro')
      geom_costum_score=sensitivity*specificity

      return {"Training Accuracy scores": results['train_accuracy'],
              "Mean Training Accuracy": results['train_accuracy'].mean()*100,
              "Training Precision scores": results['train_precision'],
              "Mean Training Precision": results['train_precision'].mean(),
              "Training Recall scores": results['train_recall'],
              "Mean Training Recall": results['train_recall'].mean(),
              "Training F1 scores": results['train_f1'],
              "Mean Training F1 Score": results['train_f1'].mean(),
              "Validation Accuracy scores": results['test_accuracy'],
              "Mean Validation Accuracy": results['test_accuracy'].mean()*100,
              "Validation Precision scores": results['test_precision'],
              "Mean Validation Precision": results['test_precision'].mean(),
              "Validation Recall scores": results['test_recall'],
              "Mean Validation Recall": results['test_recall'].mean(),
              "Validation F1 scores": results['test_f1'],
              "Mean Validation F1 Score": results['test_f1'].mean(),
              "geometric_mean_score":geom_mean_score,
              "Test_accuracy_score":test_accuracy,
              "Y_pred":y_pred,
              "geom_costum_score":geom_costum_score
      }

#'Function to plot a grouped bar chart showing the training and validation results of the ML model in each fold after applying K-fold cross-validation.
def plot_result(x_label, y_label, plot_title, X_train, Y_train,geometric_data):
   #x_label name algorithm
   #y_label: str, Name of metric being visualized e.g 'Accuracy'

        
        # Set size of plot
        plt.figure(figsize=(12,6))
        labels = ["1st Fold", "2nd Fold", "3rd Fold", "4th Fold", "5th Fold"]
        X_axis = np.arange(len(labels))
        ax = plt.gca()
        plt.ylim(0.40000, 1)
        plt.bar(X_axis+0.0, X_train, 0.2, color='blue', label='Training')
        plt.bar(X_axis+0.2, Y_train, 0.2, color='red', label='Validation')
        plt.bar(X_axis+0.4, geometric_data, 0.2, color='black', label='Geometric')
        plt.title(plot_title, fontsize=30)
        plt.xticks(X_axis, labels)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.show()

#MODEL for Naive Bayes
gnb = GaussianNB()
kf = KFold(n_splits=5)

#decision__result with callback functon
results= cross_validation(gnb, X_train_minmax, Y_train, kf)

#MEAN 
Training_acc=results["Mean Training Accuracy"]
Validation_Acc=results["Mean Validation Accuracy"]

#PREDICTION ACCURACY WITH X_TEST && Y_TEST
pred_accuracy=results['Test_accuracy_score']
pred_accuracy
#Geometric Mean score
Geom =results['geometric_mean_score']
print(f'TRAINING ACCURACY :{Training_acc}'"\n",f'VALIDATION ACCURACY :{Validation_Acc}'"\n",f'PREDICTION ACCURACY WITH TEST DATA:{pred_accuracy}'"\n",F'GEOMETRIC_MEAN_SCORE:{Geom}'"\n",F'GEOMETRIC_COSTUM_SCORE:{results["geom_costum_score"]}')


#prediction values
#print(results['Y_pred'])

#PLOT NAVE BAYES 
model_name = "Naves Bayes"

plot_result(model_name,"Accuracy","Accuracy scores in 5 Folds",results["Training Accuracy scores"],results["Validation Accuracy scores"],results['geom_costum_score'])

print(results['Y_pred'])

C=np.arange(1,202,5).tolist()


C=np.arange(1,202,5).tolist()

#Hyperparameter C calculation
kernels=['linear']
param_grid = {'C': C,'kernel': kernels}

#input normalization data 
'''X_train_minmax'''
#target output
'''Y_train'''
svc = svm.SVC()
#svc.cv_results_.keys()
#svc = svm.SVC(kernel=kernel).fit(X, y)
#plotSVC(‘kernel=’ + str(kernel))
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.25,random_state=42)
grid = GridSearchCV(svc, param_grid=param_grid,scoring='accuracy',refit=True ,cv=5,verbose=3)
grid.fit(X_train_minmax,Y_train)
print(grid.best_score_,grid.best_params_,grid.best_estimator_)
#print(grid.cv_results_)
#print(grid.cv_results_.keys())
#print("Training Accuracy scores:",grid.cv_results_['mean_train_score'].mean()*100)
print("Testing Accuracy scores:",grid.cv_results_['mean_test_score'].mean()*100)



#CGgrid = np.logspace(-15,15,num=10,base=2)
#param_grid_svm = {'C': CGgrid ,
                  #'gamma': CGgrid}
#grid_search_svm = GridSearchCV(SVC(), param_grid_svm, cv=5, scoring='accuracy', n_jobs=-1)
#grid_search_svm.fit(X, y)

#print(grid_search_svm.best_score_)     
#print(grid_search_svm.best_params_)

#Gamma me step 0.5 sto diastima 10
gamma=np.arange(0.5, 10.5, 0.5).tolist()
kernels=['rbf']
param_grid = {'C':[1.0] ,'gamma':gamma,'kernel': kernels}


  #svc = svm.SVC(kernel=kernel).fit(X, y)
  #plotSVC(‘kernel=’ + str(kernel))
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.25,random_state=42)
grid = GridSearchCV(svc,param_grid,scoring='accuracy',refit=True,cv=5,verbose=3)
grid.fit(X_train_minmax,Y_train)
print(grid.best_score_,grid.best_params_,grid.best_estimator_)
#print(grid.cv_results_)
#print(grid.cv_results_.keys())
#print("Training Accuracy scores:",grid.cv_results_['mean_train_score'].mean()*100)
print("Testing Accuracy scores:",grid.cv_results_['mean_test_score'].mean()*100)

#SVM MODEL WITH BEST PARAMETERS


#MODEL for SVM
svc = svm.SVC(kernel='rbf',C=1.0,gamma=2.5)
kf = KFold(n_splits=5)

#decision__result with callback functon
svm_results= cross_validation(svc, X_train_minmax, Y_train, kf)

#MEAN 
Training_acc=svm_results["Mean Training Accuracy"]
Validation_Acc=svm_results["Mean Validation Accuracy"]

#PREDICTION ACCURACY WITH X_TEST && Y_TEST
pred_accuracy=svm_results['Test_accuracy_score']
pred_accuracy
#Geometric Mean score
Geom =svm_results['geometric_mean_score']
geom_list=[Geom]*5

print(f'TRAINING ACCURACY :{Training_acc}'"\n",f'VALIDATION ACCURACY :{Validation_Acc}'"\n",f'PREDICTION ACCURACY WITH TEST DATA:{pred_accuracy}'"\n",F'GEOMETRIC_MEAN_SCORE:{Geom}'"\n",F'GEOMETRIC_COSTUM_SCORE:{svm_results["geom_costum_score"]}')

#PLOT SVM
model_name = "Support Vector Machine"

plot_result(model_name,"Accuracy","Accuracy scores in 5 Folds",svm_results["Training Accuracy scores"],svm_results["Validation Accuracy scores"],svm_results['geom_costum_score'])

print(svm_results['Y_pred'])

#create new a knn model
knn2 = KNeighborsClassifier()
#create a dictionary of all values we want to test for n_neighbors
param_grid = {"n_neighbors": np.arange(3, 16)}
#use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(knn2, param_grid, cv=5)
#fit model to data
knn_gscv.fit(X_train_minmax, Y_train)

print(knn_gscv.best_params_,knn_gscv.best_score_)

#create a new KNN model
knn_cv = KNeighborsClassifier(n_neighbors=12)
#train model with cv of 5 
cv_scores = cross_val_score(knn_cv, X_train_minmax, Y_train, cv=5)
#print each cv score (accuracy) and average them
print(cv_scores)
print("cv_scores mean:{}".format(np.mean(cv_scores)))

#MODEL for k-neigh
knn_cv = KNeighborsClassifier(n_neighbors=14)
kf = KFold(n_splits=5)

#decision__result with callback functon
knn_results= cross_validation(knn_cv, X_train_minmax, Y_train, kf)

#MEAN 
Training_acc=knn_results["Mean Training Accuracy"]
Validation_Acc=knn_results["Mean Validation Accuracy"]

#PREDICTION ACCURACY WITH X_TEST && Y_TEST
pred_accuracy=knn_results['Test_accuracy_score']
pred_accuracy
#Geometric Mean score
Geom =knn_results['geometric_mean_score']
geom_list=[Geom]*5

print(f'TRAINING ACCURACY :{Training_acc}'"\n",f'VALIDATION ACCURACY :{Validation_Acc}'"\n",f'PREDICTION ACCURACY WITH TEST DATA:{pred_accuracy}'"\n",F'GEOMETRIC_MEAN_SCORE:{Geom}'"\n",F'GEOMETRIC_COSTUM_SCORE:{knn_results["geom_costum_score"]}')

#PLOT K-NEIGH
model_name = "k-Nearest-Neighbors"

plot_result(model_name,"Accuracy","Accuracy scores in 5 Folds",knn_results["Training Accuracy scores"],knn_results["Validation Accuracy scores"],knn_results['geom_costum_score'])

print(knn_results['Y_pred'])

'''Geometric Mean = sqrt (Sensitivity * Specificity)
sensitivity = sklearn.recall_score(Y_tr , pred)
specificity = sklearn.recall_score(np.logical_not(true) , np.logical_not(pred)) '''
age,tb_bilirubin,DB_Bilirubin,Alkaline_Phosphotase,Sgot,Protiens,Albumin,Albumin_Globulin = np.mean(X_data['Age']), np.mean(X_data['Tb_Bilirubin']), np.mean(X_data['DB_Bilirubin']), np.mean(X_data['Alkaline_Phosphotase']), np.mean(X_data['Sgot']),np.mean(X_data['Protiens']), np.mean(X_data['Albumin']),np.mean(X_data['Albumin_Globulin'])

#ERWTIMA 5
X_data

#ERWTIMA 5 
#STUDENT T-TEST

age,Tb_Bilirubin,DB_Bilirubin,Alkaline_Phosphotase,sgpt,sgot,Protiens,Albumin,Albumin_Globulin=csv.Age,csv.Tb_Bilirubin,csv.DB_Bilirubin,csv.Alkaline_Phosphotase,csv['sgpt '],csv.Sgot,csv.Protiens,csv.Albumin,csv.Albumin_Globulin
print(shapiro(age))
print(shapiro(Tb_Bilirubin))
print(shapiro(DB_Bilirubin))
print(shapiro(Alkaline_Phosphotase))
print(shapiro(sgpt))
print(shapiro(sgot))
print(shapiro(Protiens))
print(shapiro(Albumin))
print(shapiro(Albumin_Globulin))

skewed = ['Alkaline_Phosphotase', 'DB_Bilirubin', 'Tb_Bilirubin', 'sgpt ', 'Sgot']
X_data =X_data.drop(columns=['Age' , 'Gender' , 'Protiens','Albumin','Albumin_Globulin'])
X_data

#Create Input && output data 
X_data = csv.drop('Disease',axis=1)
Y_data = csv['Disease']

#split dataset to train and test data 
X_train,X_test,Y_train,Y_test =train_test_split(X_data,Y_data,test_size=0.25)

def preprocessing(X_train,Y_train,X_test,Y_test,type="MinMax"):

            #NORMALIZATION#
    if type == "Normalization":
        X_train_normalized = tf.keras.utils.normalize(X_train)
        X_test_normalized = tf.keras.utils.normalize(X_test)
        return X_train_normalized,Y_train,X_test_normalized,Y_test

            #NORM-WITH MINMAX#
    elif type == "MinMax":
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_train_minmax = scaler.fit_transform(X_train)
        X_test_minmax = scaler.fit_transform(X_test)
        return X_train_minmax , Y_train ,X_test_minmax ,Y_test

X_train_minmax,Y_train,X_test_minmax,Y_test=preprocessing(X_train,Y_train,X_test,Y_test,type="MinMax")
print(X_train_minmax.shape,X_test_minmax.shape,Y_train.shape,Y_test.shape)

#SVM MODEL WITH BEST PARAMETERS

#MODEL for SVM
svc = svm.SVC(kernel='rbf',C=1.0,gamma=2.5)
kf = KFold(n_splits=5)

#decision__result with callback functon
svm_results= cross_validation(svc, X_train_minmax, Y_train, kf)

#MEAN 
Training_acc=svm_results["Mean Training Accuracy"]
Validation_Acc=svm_results["Mean Validation Accuracy"]

#PREDICTION ACCURACY WITH X_TEST && Y_TEST
pred_accuracy=svm_results['Test_accuracy_score']
pred_accuracy
#Geometric Mean score
Geom =svm_results['geometric_mean_score']
geom_list=[Geom]*5

print(f'TRAINING ACCURACY :{Training_acc}'"\n",f'VALIDATION ACCURACY :{Validation_Acc}'"\n",f'PREDICTION ACCURACY WITH TEST DATA:{pred_accuracy}'"\n",F'GEOMETRIC_MEAN_SCORE:{Geom}'"\n",F'GEOMETRIC_COSTUM_SCORE:{svm_results["geom_costum_score"]}')

#PLOT SVM
model_name = "Support Vector Machine"

plot_result(model_name,"Accuracy","Accuracy scores in 5 Folds",svm_results["Training Accuracy scores"],svm_results["Validation Accuracy scores"],svm_results['geom_costum_score'])

print(svm_results['Y_pred'])



