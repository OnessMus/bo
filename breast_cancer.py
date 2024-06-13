import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

#Importing
data = pd.read_csv('data.csv')
#print(data.head())

#print(data.columns)
#print(data.shape)

#print(data.diagnosis.value_counts())

#data types
#print(data.dtypes)

#Identifying unique values
#print(data.nunique())

#Checking sum of NULL values
#print(data.isnull().sum())

#droping unnamed 32 and the column ID
data.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)
#print(data[data.isnull().any(axis=1)])

#Viewing data statistics
#print(data.describe)

#Data Visualization
#Finding out the correlation between the features
non_numeric_columns = data.select_dtypes(include=['object']).columns
data_numeric = data.drop(columns=non_numeric_columns)
corr = data_numeric.corr()
#print(corr.shape)

#Plotting the heatmap of the correlation between features
#plt.figure(figsize=(20,20))
#sns.heatmap(corr, cbar=True, square= True, fmt='.1f', annot=True, annot_kws={'size':15}, cmap='Greens')
#plt.show()

#Analysing the target variable
#plt.title('Count of cancer type')
#sns.countplot(data['diagnosis'])
#plt.xlabel('Cancer lethality')
#plt.ylabel('Count')
#plt.show()

#Plotting correlation between diagnosis and radius
#plt.figure(figsize=(10,5))
#plt.subplot(1,2,1)
#sns.boxplot(x="diagnosis", y="radius_mean", data=data)
#plt.subplot(1,2,2)
#sns.violinplot(x="diagnosis", y="radius_mean", data=data)
#plt.show()

#Plotting correlation between diagnosis and concatinity
#plt.figure(figsize=(10,5))
#plt.subplot(1,2,1)
#sns.boxplot(x="diagnosis", y="concavity_mean", data=data)
#plt.subplot(1,2,2)
#sns.violinplot(x="diagnosis", y="concavity_mean", data=data)
#plt.show()

#Distribution density plot KDE
#g = sns.FacetGrid(data, hue="diagnosis", height=6)
#g.map(sns.kdeplot, "radius_mean").add_legend()
#plt.show()

#Plotting the distribution of the mean radius
#sns.stripplot(x="diagnosis", y="radius_mean", data=data, jitter=True, edgecolor="gray")
#plt.show()

#Plotting bivariate relations between each pair of the features
#sns.pairplot(data, hue="diagnosis", vars=["radius_mean", "concavity_mean", "smoothness_mean", "texture_mean"])
#plt.show()

#MODELING
#Splitting target variable and independent variables
X = data.drop(['diagnosis'], axis = 1)
y = data['diagnosis']

#Splitting the data into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
#print("Size of training set:", X_train.shape)
#print("Size of test set:", X_test.shape)

#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
pipeline.fit(X_train, y_train)

#Prediction on the test data
y_pred = pipeline.predict(X_test)

#Calculating the accuracy
acc_logreg = round(metrics.accuracy_score(y_test, y_pred) * 100, 2)
#print('Accuracy of the Logistic Regression  model : ', acc_logreg)

#GAUSSIAN NAIVE BAYES
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)

#prediction on the test set
y_pred = model.predict(X_test)

#Calculating accuracy
acc_nb = round(metrics.accuracy_score(y_test, y_pred) * 100, 2)
#print('Accuracy of Gaussian Naive Bayes model : ', acc_nb)

#DECISION TREE
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()

#Hyperparameter optimization
parameters = {'max_features': ['log2', 'sqrt', None],
              'criterion': ['entropy', 'gini'],
              'max_depth': [2,3,5,10,50],
              'min_samples_split': [2,3,50,100],
              'min_samples_leaf': [1,5,8,10]
              }
#Run the Grid search
grid_obj = GridSearchCV(clf, parameters)
grid_obj = grid_obj.fit(X_train, y_train)

#Set clf to the best combination of parameters
clf = grid_obj.best_estimator_

#Train the model using the training set
clf.fit(X_train, y_train)

#prediction on the test
y_pred = clf.predict(X_test)

#Calculating the accuracy
acc_dt = round(metrics.accuracy_score(y_test, y_pred) * 100, 2)
print('Accuracy of the Decision Tree Model : ', acc_dt)



#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()

#Hyperparameter optimization
parameters = {'n_estimators': [4,6,9,10,15],
              'max_features': ['log2', 'sqrt', None],
              'criterion': ['entropy', 'gini'],
              'max_depth': [2,3,5,10],
              'min_samples_split': [2,3,5],
              'min_samples_leaf': [1,5,8]
              }
#Run the Grid search
grid_obj = GridSearchCV(rf, parameters)
grid_obj = grid_obj.fit(X_train, y_train)

#Set rf to the best combination of parameters
rf = grid_obj.best_estimator_

#Train the model using the training set
rf.fit(X_train, y_train)

#prediction on the test
y_pred = rf.predict(X_test)

#Calculating the accuracy
acc_rf = round(metrics.accuracy_score(y_test, y_pred) * 100, 2)
print('Accuracy of the Decision Tree Model : ', acc_rf)



#SUPPORT VECTOR MACHINE
#SVM Classifier
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn import svm
svc = svm.SVC()

parameters = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
  ]
#Run grid search
grid_obj = GridSearchCV(svc, parameters)
grid_obj = grid_obj.fit(X_train, y_train)

#Set the svc to the best combination of parameters
svc = grid_obj.best_estimator_

#Train the model using the training sets
svc.fit(X_train, y_train)

#Prediction on the test data
y_pred = svc.predict(X_test)

#Calculating the accuracy
acc_svm = round(metrics.accuracy_score(y_test, y_pred) * 100, 2)
print('Accuracy of the SVM model : ', acc_svm)



#K -Nearest Neighbours

# Import library of KNeighborsClassifier model
from sklearn.neighbors import KNeighborsClassifier

# Create a KNN Classifier
knn = KNeighborsClassifier()

# Hyperparameter Optimization
parameters = {'n_neighbors': [3, 4, 5, 10], 
              'weights': ['uniform', 'distance'],
              'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],
              'leaf_size' : [10, 20, 30, 50]
             }

# Run the grid search
grid_obj = GridSearchCV(knn, parameters)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the knn to the best combination of parameters
knn = grid_obj.best_estimator_

# Train the model using the training sets 
knn.fit(X_train,y_train)

# Prediction on test data
y_pred = knn.predict(X_test)

# Calculating the accuracy
acc_knn = round( metrics.accuracy_score(y_test, y_pred) * 100, 2 )
print( 'Accuracy of KNN model : ', acc_knn )


#EVALUATION AND COMPARISON OF ALL MODELS
models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Naive Bayes', 'Decision Tree', 'Random Forest', 'Support Vector Machines', 
              'K - Nearest Neighbors'],
    'Score': [acc_logreg, acc_nb, acc_dt, acc_rf, acc_svm, acc_knn]})
models.sort_values(by='Score', ascending=False)
    
    



