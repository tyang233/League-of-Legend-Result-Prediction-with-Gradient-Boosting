import pandas as pd  
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
import numpy as np 
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt 
from sklearn.metrics import roc_curve 

#define the accuracy function
def accuracy(data):
    
    target='winner'
    IDcol= 'gameId'  
    time='creationTime'
    season='seasonId'
    data['winner'].value_counts()  
    
    x_columns= [x for x in data.columns if x not in [target, IDcol,time,season]]  
    X= data[x_columns]
    y= data['winner']  
    #apply the gradient boost classifier
    gbm4= GradientBoostingClassifier(learning_rate=0.005,n_estimators=700,max_depth=17, min_samples_leaf = 80, min_samples_split =1500,max_features=22, subsample=0.7, random_state=10)  
    gbm4.fit(X,y)  
    y_pred= gbm4.predict(X) 
    y_predprob= gbm4.predict_proba(X)[:,1] 
    a=metrics.accuracy_score(y.values, y_pred)
    b=metrics.roc_auc_score(y, y_predprob)
        
    return a,b

#define the main function
def main():
    data = pd.read_csv('games.csv',index_col=0)
    data['winner']=data['winner']-1
    #reform the data as two dataframes
    mylist = ['t1_ban1', 't1_ban2','t1_ban3','t1_ban4','t1_ban5','t2_ban1', 't2_ban2','t2_ban3','t2_ban4','t2_ban5']
    for i in mylist:
        data = data[data[i]!=-1]
    data1 = data.drop(['t1_towerKills','t1_inhibitorKills','t1_baronKills','t1_dragonKills','t1_riftHeraldKills',
                       't2_towerKills','t2_inhibitorKills','t2_baronKills','t2_dragonKills','t2_riftHeraldKills',
                       'firstBlood','firstTower', 'firstInhibitor', 'firstBaron', 'firstDragon', 'firstRiftHerald'], 1)

    data1 = pd.get_dummies(data1, columns=['t1_ban1', 't1_ban2','t1_ban3','t1_ban4','t1_ban5',
        't1_champ1id','t1_champ2id','t1_champ3id','t1_champ4id','t1_champ5id',
        't2_ban1', 't2_ban2','t2_ban3','t2_ban4','t2_ban5',
        't2_champ1id','t2_champ2id','t2_champ3id','t2_champ4id','t2_champ5id',
        't1_champ1_sum1','t1_champ1_sum2','t1_champ2_sum1','t1_champ2_sum2','t1_champ3_sum1','t1_champ3_sum2',
                     't1_champ4_sum1','t1_champ4_sum2','t1_champ5_sum1','t1_champ5_sum2','t2_champ1_sum1','t2_champ1_sum2',
                     't2_champ2_sum1','t2_champ2_sum2','t2_champ3_sum1','t2_champ3_sum2','t2_champ4_sum1','t2_champ4_sum2',
                     't2_champ5_sum1','t2_champ5_sum2'
        ])

    data2 = pd.get_dummies(data, columns=['t1_ban1', 't1_ban2','t1_ban3','t1_ban4','t1_ban5',
        't1_champ1id','t1_champ2id','t1_champ3id','t1_champ4id','t1_champ5id',
        't2_ban1', 't2_ban2','t2_ban3','t2_ban4','t2_ban5',
        't2_champ1id','t2_champ2id','t2_champ3id','t2_champ4id','t2_champ5id',
        't1_champ1_sum1','t1_champ1_sum2','t1_champ2_sum1','t1_champ2_sum2','t1_champ3_sum1','t1_champ3_sum2',
                     't1_champ4_sum1','t1_champ4_sum2','t1_champ5_sum1','t1_champ5_sum2','t2_champ1_sum1','t2_champ1_sum2',
                     't2_champ2_sum1','t2_champ2_sum2','t2_champ3_sum1','t2_champ3_sum2','t2_champ4_sum1','t2_champ4_sum2',
                     't2_champ5_sum1','t2_champ5_sum2',
                     'firstBlood','firstTower', 'firstInhibitor', 'firstBaron', 'firstDragon', 'firstRiftHerald'
        ])

    target='winner'
    IDcol= 'gameId'  
    time='creationTime'
    season='seasonId'

    #train the pregame feature dataset
    train= data1
    a,b= accuracy(train)
    print('Accuracy for pre_game_feature: %.4g' % a)
    print("AUC Score (Train) for pre_game_feature: %f" % b)
    x_columns= [x for x in train.columns if x not in [target, IDcol,time,season]]  
    X= np.array(train[x_columns])
    y= np.array(train['winner'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.33, random_state=42)
    
    grd = GradientBoostingClassifier(learning_rate=0.005,n_estimators=700,max_depth=17, min_samples_leaf = 80, min_samples_split =1500,max_features=22, subsample=0.7, random_state=10)
    grd.fit(X_train, y_train)
    

    y_pred_grd = grd.predict_proba(X_test)[:, 1]
    fpr_grd, tpr_grd, _ = roc_curve(y_test, y_pred_grd)
    
    # train the dataset including all the game features
    train2= data2
    c,d= accuracy(train2)
    print('Accuracy for all_game_feature: %.4g' % c)
    print("AUC Score (Train) for all_game_feature: %f" % d)
    x_columns2= [x for x in train2.columns if x not in [target, IDcol,time,season]]  
    X2= np.array(train2[x_columns2])
    y2= np.array(train2['winner'])
    
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, 
                                                        test_size=0.33, random_state=42)
    grd2 = GradientBoostingClassifier(learning_rate=0.005,n_estimators=700,max_depth=17, min_samples_leaf = 80, min_samples_split =1500,max_features=22, subsample=0.7, random_state=10)
    grd2.fit(X_train2, y_train2)
    
    y_pred_grd2 = grd2.predict_proba(X_test2)[:, 1]
    fpr_grd2, tpr_grd2, _ = roc_curve(y_test2, y_pred_grd2)

    #Plot the roc curve
    plt.figure(1)
    plt.plot(fpr_grd, tpr_grd, label='GBT for pregame_feature')
    plt.plot(fpr_grd2, tpr_grd2, label='GBT for all game_feature')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC curve')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend(loc='best')
    plt.show()
    
main()
