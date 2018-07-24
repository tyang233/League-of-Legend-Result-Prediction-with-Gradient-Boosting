
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import json

#define the main function
def main():
    #read the data into dataframe
    data = pd.read_csv('games.csv')
    mylist = ['t1_ban1', 't1_ban2','t1_ban3','t1_ban4','t1_ban5','t2_ban1', 't2_ban2','t2_ban3','t2_ban4','t2_ban5']
    for i in mylist:
        data = data[data[i]!=-1]
    #create a new dataframe that drop all the first feateures
    data1 = data.drop(['firstBlood','firstTower', 'firstInhibitor', 'firstBaron', 'firstDragon', 'firstRiftHerald'], 1)
    num_records = len(data)
    num_col = len(data.columns)
    #print the size of the dataset
    print("The number of records is: %d"%(num_records))
    print("The number of columns is: %d"%(num_col))
    #create the visualizaiton
    myboxplot(data)
    mybarchart(data)

#define the boxplot function
def myboxplot(data):
    data_Clean = data.replace([0,1,2],['neither','blue','red'])
    plt.figure()
    data_Clean.boxplot(column = 'gameDuration',by = 'firstTower')
    plt.show()
    plt.savefig('boxplot.png',bbox_inches='tight')

#define the barchart function
def mybarchart(data):
    data_Clean = data.replace([0,1,2],['neither','blue','red'])
    firsts = ['firstBlood','firstTower', 'firstInhibitor', 'firstBaron', 'firstDragon', 'firstRiftHerald']
    firstTotals = data_Clean[firsts].apply(pd.value_counts)
    newIndex = ['blue','red','neither']
    firstSort = firstTotals.reindex(index=newIndex)
    #set the plotting colors
    plotColors = ['red','blue','grey']
    firstLabels = ['First Blood','First Tower', 'First Inhibitor', 'First Baron', 'First Dragon', 'First Rift Herald']
    nrows, ncols = 2,3
    fig = plt.figure(figsize=(15,10))
    for i in range(1,7):
        ax = fig.add_subplot(nrows,ncols,i)
        sns.barplot(x=firstSort.index,y=firstSort[firstSort.columns[i-1]],palette=plotColors)
        ax.set_ylabel('Count')
        ax.yaxis.set_ticklabels([])
        ax.set_title(firstLabels[i-1])

    plt.savefig("Histogram_compairision")
    plt.show()

main()