
import statistics
import numpy as np
import pandas as pd
def knn(train,test,k):
    result = []
    for data in test:
        res = (np.sqrt(np.sum((train[:,:-1]-data)*(train[:,:-1]-data),axis = 1)))
        res = np.array(sorted([[i,j] for i,j in enumerate(res)],key = lambda x: x[1]))
        res = res.astype('int64')
        val = (np.take(train[:,-1],res[:k,0],axis =0))
        output_class = 0 if np.sum(val == 0)>=np.sum(val ==1) else 1
        result.append(output_class)
    return result    
def main():
    train = pd.read_csv("project2.csv").to_numpy()
    test = pd.read_csv("project2_test.csv").to_numpy()
    for k in range(train.shape[1]-1):
        train[:,k]=((train[:,k]-statistics.mean(train[:,k]))/np.sqrt(statistics.variance(train[:,k])))
    #print(train[:5])  
    k = int(input("Enter the value of k(-1 default): "))
    k = np.sqrt(train.shape[0]) if k==-1 else k  
    result = knn(train,test,k)
    print("Output classes: ",end = "")
    for v in result:
        print(v,end=' ')
    print()    
if __name__ == "__main__":
    main()