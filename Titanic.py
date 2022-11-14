import pandas as pd
import numpy as np
import matplotlib
# matplotlib.use('MacOSX')
from matplotlib import pyplot as plt
import seaborn as sns
import sklearn

path = "/Users/mateusz/Downloads/train.csv"
path_1 = "/Users/mateusz/Downloads/test.csv"
train = pd.read_csv(path)
test = pd.read_csv(path_1)

train['train_test'] = 1
test['train_test'] = 0
test['Survived'] = np.NaN
all_data = pd.concat([train, test])

te = test[['PassengerId']]

print(train.columns)

numerical = train[["Age", "SibSp", "Parch", "Fare"]]
categorical = train[["Survived", "Pclass", "Sex", "Ticket", "Cabin", "Embarked"]]

# for i in numerical.columns:
#     plt.hist(numerical[i])
#     plt.title(i)
#     plt.show()
print(numerical.corr())
print(pd.pivot_table(train, index= 'Survived', values= numerical))

# for i in categorical.columns:
#     sns.barplot(x=categorical[i].value_counts().index, y=categorical[i].value_counts()).set(title=i)
#     plt.show()

print(pd.pivot_table(train,index= 'Survived', columns= 'Pclass', values= 'Ticket', aggfunc= 'count'))
print(pd.pivot_table(train,index= 'Survived', columns= 'Sex', values= 'Ticket', aggfunc= 'count'))
print(pd.pivot_table(train,index= 'Survived', columns= 'Embarked', values= 'Ticket', aggfunc= 'count'))


# Function that creates a column named cabin_multi
# Also fills it with 0 if no function, otherwise fills it with number of seperate strings
train['cabin_multi']= train.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))
print(pd.pivot_table(train, index='Survived', columns='cabin_multi', values='Ticket', aggfunc='count'))

# Function that creates a column named cabin_first_letter
# Also fills it with first letter of the string from the cabin they where in
train['cabin_first_letter'] = train.Cabin.apply(lambda x: str(x)[0])
print(pd.pivot_table(train,index='Survived',columns='cabin_first_letter',values='Ticket',aggfunc='count'))

train['numeric_number'] = train.Ticket.apply(lambda x:1 if x.isnumeric() else 0)
print(pd.pivot_table(train,index='Survived',columns='numeric_number',values='Ticket',aggfunc='count'))

train['ticket_letters'] = train.Ticket.apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.', '').replace
('/', '').lower() if len(x.split(' ')[:-1]) >0 else 0)
print(pd.pivot_table(train,index='Survived',columns='ticket_letters',values='Ticket',aggfunc='count'))

# converting N/A to the mean values
train.Age = train.Age.fillna(train.Age.mean())
train.Fare = train.Fare.fillna(train.Fare.median())

#drops N/A from embarked
train.dropna(subset=['Embarked'],inplace = True)

train.Pclass = train.Pclass.astype(str)



# we transform all data including test data to look the same

all_data['cabin_multi']= all_data.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))
all_data['cabin_first_letter'] = all_data.Cabin.apply(lambda x: str(x)[0])
all_data['numeric_number'] = all_data.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)
all_data['ticket_letters'] = all_data.Ticket.apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.', '').replace
('/', '').lower() if len(x.split(' ')[:-1]) >0 else 0)
all_data.Age = all_data.Age.fillna(all_data.Age.mean())
all_data.Fare = all_data.Fare.fillna(all_data.Fare.median())
all_data.dropna(subset=['Embarked'],inplace = True)
all_data.Pclass = all_data.Pclass.astype(str)
all_data['norm_fare'] = np.log(all_data.Fare+1)

all_dummies = pd.get_dummies(all_data[["Age", "SibSp", "Parch", 'norm_fare', "Survived",
                                       "Pclass", "Sex", "Embarked", 'cabin_multi', 'train_test']])
x_train = all_dummies[all_dummies.train_test == 1].drop(['train_test'], axis=1)
test = all_dummies[all_dummies.train_test == 0].drop(['train_test'], axis=1)


from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
all_dummies_scaled = all_dummies.copy()
all_dummies_scaled[["Age", "SibSp", "Parch", 'norm_fare']] = \
    scale.fit_transform(all_dummies_scaled[["Age", "SibSp", "Parch", 'norm_fare']])

x_train_scaled = all_dummies_scaled[all_dummies_scaled.train_test == 1].drop(['train_test'], axis=1)
x_test_scaled = all_dummies_scaled[all_dummies_scaled.train_test == 0].drop(['train_test'], axis=1)
y_train = all_data[all_data.train_test == 1].Survived
x_train_scaled = x_train_scaled.drop(['Survived'], axis=1)
x_test_scaled = x_test_scaled.drop(['Survived'], axis=1)

from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from xgboost import XGBRegressor

gnb = GaussianNB()
cvg = cross_val_score(gnb, x_train_scaled, y_train, cv=5)

lr = LogisticRegression(max_iter = 2000)
cvl = cross_val_score(lr, x_train_scaled, y_train, cv=5)

dt = tree.DecisionTreeClassifier(random_state=1)
cvd = cross_val_score(dt, x_train_scaled, y_train, cv=5)
cvd_1 = cross_val_score(dt, x_train, y_train, cv=5)

knn = KNeighborsClassifier()
cvk = cross_val_score(knn, x_train_scaled, y_train, cv=5)
cvk_1 = cross_val_score(knn, x_train, y_train, cv=5)

rft = RandomForestClassifier(random_state=1)
cvt = cross_val_score(rft, x_train_scaled, y_train, cv=5)
cvt_1 = cross_val_score(rft, x_train, y_train, cv=5)

svc = SVC(probability=True)
cvs = cross_val_score(svc, x_train_scaled, y_train, cv=5)

xgc = XGBClassifier(random_state=1)
cvx = cross_val_score(xgc, x_train_scaled, y_train, cv=5)

xgb = XGBRegressor()
cvx_1 = cross_val_score(xgb, x_train_scaled, y_train, cv=5)


print("GaussianNB precision is ", cvg.mean())
print("LogisticRegression precision was ", cvl.mean())
print("DecisionTreeClassifier precision is ", cvd.mean())
print("DecisionTreeClassifier norm precision is ", cvd_1.mean())
print("KNeighborsClassifier precision is ", cvk.mean())
print("KNeighborsClassifier norm precision is ", cvk_1.mean())
print("RandomForestClassifier precision is ", cvt.mean())
print("RandomForestClassifier norm precision is ", cvt_1.mean())
print("SVC precision is ", cvs.mean())
print("XGBClassifier precision is ", cvx.mean())
print("XGBRegressor precision is ", cvx_1.mean())

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


def clf_performance(classifier, model_name):
    print(model_name)
    print("Best Score: ", str(classifier.best_score_))
    print("Best Parameters: ", str(classifier.best_params_))

lr = LogisticRegression()
param_grid = {'max_iter' : [2000],
              'penalty' : ['l1', 'l2'],
              'C' : np.logspace(-4, 4, 20),
              'solver' : ['liblinear']}
clf_lr = GridSearchCV(lr, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
best_clf_lr = clf_lr.fit(x_train_scaled, y_train)
clf_performance(best_clf_lr,'Logistic Regression')

knn = KNeighborsClassifier()
param_grid = {'n_neighbors' : [3,5,7,9],
              'weights' : ['uniform', 'distance'],
              'algorithm' : ['auto', 'ball_tree','kd_tree'],
              'p' : [1,2]}
clf_knn = GridSearchCV(knn, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
best_clf_knn = clf_knn.fit(x_train_scaled,y_train)
clf_performance(best_clf_knn,'KNN')

svc = SVC(probability = True)
param_grid = tuned_parameters = [{'kernel': ['rbf'], 'gamma': [.1,.5,1,2,5,10],
                                  'C': [.1, 1, 10, 100]},
                                 {'kernel': ['linear'], 'C': [.1, 1, 10, 100]},
                                 {'kernel': ['poly'], 'degree' : [2,3,4,5], 'C': [.1, 1, 10, 100]}]
clf_svc = GridSearchCV(svc, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
best_clf_svc = clf_svc.fit(x_train_scaled,y_train)
clf_performance(best_clf_svc,'SVC')

rf = RandomForestClassifier(random_state=1)
param_grid = {'n_estimators': [400, 450],
              'criterion': ['gini', 'entropy'],
              'bootstrap': [True],
              'max_depth': [15, 20, 25],
              'max_features': ['sqrt', 10],
              'min_samples_leaf': [2, 3],
              'min_samples_split': [2, 3]}

clf_rf = GridSearchCV(rf, param_grid=param_grid, cv=5, verbose=True, n_jobs=-1)
best_clf_rf = clf_rf.fit(x_train_scaled, y_train)
clf_performance(best_clf_rf, 'Random Forest')

xgb = XGBClassifier(random_state = 1)
param_grid = {
    'n_estimators': [450,500,550],
    'colsample_bytree': [0.65,0.75],
    'max_depth': [None],
    'reg_alpha': [1],
    'reg_lambda': [2, 5, 10],
    'subsample': [0.6, .65],
    'learning_rate':[0.4,0.5],
    'gamma':[.5,1,2],
    'min_child_weight':[0.01,0.03],
    'sampling_method': ['uniform']
}

clf_xgb = GridSearchCV(xgb, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
best_clf_xgb = clf_xgb.fit(x_train_scaled,y_train)
clf_performance(best_clf_xgb,'XGB')

best_lr = best_clf_lr.best_estimator_
best_knn = best_clf_knn.best_estimator_
best_svc = best_clf_svc.best_estimator_
best_rf = best_clf_rf.best_estimator_
best_xgb = best_clf_xgb.best_estimator_




# xgb = XGBRegressor(random_state = 1)
# param_grid = {
#     'n_estimators': [450,500,550],
#     'colsample_bytree': [0.75,0.8,0.85],
#     'max_depth': [3,6,8,10],
#     'reg_alpha': [1],
#     'reg_lambda': [2, 5, 10],
#     'subsample': [0.55, 0.6, .65],
#     'learning_rate':[0.5],
#     'gamma':[.5,1,2],
#     'min_child_weight':[0.01],
#     'sampling_method': ['uniform']
# }
# clf_xgb = GridSearchCV(xgb, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
# best_clf_xgb = clf_xgb.fit(x_train_scaled,y_train)
# clf_performance(best_clf_xgb,'XGB')

from sklearn.ensemble import VotingClassifier

voting_clf_hard = VotingClassifier(estimators = [('knn',best_knn),('rf',best_rf),('svc',best_svc)], voting = 'hard')
voting_clf_soft = VotingClassifier(estimators = [('knn',best_knn),('rf',best_rf),('svc',best_svc)], voting = 'soft')
voting_clf_all = VotingClassifier(estimators = [('knn',best_knn),('rf',best_rf),('svc',best_svc), ('lr', best_lr)], voting = 'soft')
voting_clf_xgb = VotingClassifier(estimators = [('knn',best_knn),('rf',best_rf),('svc',best_svc), ('xgb', best_xgb),('lr', best_lr)], voting = 'soft')

print('voting_clf_hard :',cross_val_score(voting_clf_hard,x_train,y_train,cv=5))
print('voting_clf_hard mean :',cross_val_score(voting_clf_hard,x_train,y_train,cv=5).mean())

print('voting_clf_soft :',cross_val_score(voting_clf_soft,x_train,y_train,cv=5))
print('voting_clf_soft mean :',cross_val_score(voting_clf_soft,x_train,y_train,cv=5).mean())

voting_clf_hard.fit(x_train_scaled, y_train)
voting_clf_soft.fit(x_train_scaled, y_train)
voting_clf_all.fit(x_train_scaled, y_train)
voting_clf_xgb.fit(x_train_scaled, y_train)


y_hat_vc_hard = voting_clf_hard.predict(x_test_scaled).astype(int)
y_hat_rf = best_rf.predict(x_test_scaled).astype(int)
y_hat_vc_soft =voting_clf_soft.predict(x_test_scaled).astype(int)
y_hat_vc_all = voting_clf_all.predict(x_test_scaled).astype(int)
y_hat_vc_xgb = voting_clf_xgb.predict(x_test_scaled).astype(int)


final_data = {'PassengerId': te.PassengerId, 'Survived': y_hat_rf}
submission = pd.DataFrame(data=final_data)

final_data_2 = {'PassengerId': te.PassengerId, 'Survived': y_hat_vc_hard}
submission_2 = pd.DataFrame(data=final_data_2)

final_data_3 = {'PassengerId': te.PassengerId, 'Survived': y_hat_vc_soft}
submission_3 = pd.DataFrame(data=final_data_3)

final_data_4 = {'PassengerId': te.PassengerId, 'Survived': y_hat_vc_all}
submission_4 = pd.DataFrame(data=final_data_4)

final_data_5 = {'PassengerId': te.PassengerId, 'Survived': y_hat_vc_xgb}
submission_5 = pd.DataFrame(data=final_data_5)
#
# final_data_comp = {'PassengerId': te.PassengerId, 'Survived_vc_hard': y_hat_vc_hard, 'Survived_rf': y_hat_rf, 'Survived_vc_soft' : y_hat_vc_soft, 'Survived_vc_all' : y_hat_vc_all,  'Survived_vc_xgb' : y_hat_vc_xgb}
# comparison = pd.DataFrame(data=final_data_comp)

# export to excel
submission.to_csv('submission1_rf.csv', index =False)
submission_2.to_csv('submission2_rf.csv', index =False)
submission_3.to_csv('submission3_rf.csv', index =False)
submission_4.to_csv('submission4_rf.csv', index =False)
submission_5.to_csv('submission5_rf.csv', index =False)




