from collections import Counter
from os import system
import numpy as np
def key(item):
    return item[1]
def cardinality(df):
    print ("Computing measures for the chosen dataset...")
    datafilename="./datasets/"+df+".complete"
    
    if datafilename.lower().endswith('.complete') == False :
        system.exit("Dataset format unknown, please use .arff datasets")    
    
    datafile=open(datafilename)
    l0=datafile.readline()
    l0=l0.split()
    sparse = l0[1]
    if sparse[:-1] == 'SPARSE':
        sparse = True #The file is in sparse mode
    else:
        sparse = False
    
    l1=datafile.readline()
    l2=datafile.readline()
    l3=datafile.readline()
    instances=int(l1.split()[1])
    #print instances
    features=int(l2.split()[1])
    #print features
    labels=int(l3.split()[1])
    #print labels

    l4=datafile.readline()

    avg=0
    tmp=0
    dist=[]
    insts = np.zeros(labels,dtype=int)

    nwdfname="./datasets/"+df+".dsetm"
    fp=open(nwdfname, 'w')
    fp.write("Instances: "+ str(instances)+'\n')
    fp.write("Features: "+ str(features)+'\n')
    fp.write("Labels: "+ str(labels)+'\n')
    while l4 != "":
        if(l4 == ' '):
            pass
        else:
            if sparse == False:  
                label = map(int, l4.strip().split()[features+1:features+1+labels])
                #To remove the '[' ']' from the labels extraction
                dist.append(''.join(map(str, l4.strip().split()[features+1:features+1+labels])))
                #print dist en dist tenemos todas las combinacs, luego hacemos el set
                tmp = sum(label)
                insts[tmp] += 1
                avg += sum(label)
            else:
                #Sparse . find '[' and start reading until ']'
                label = map(int, l4.strip().split()[l4.strip().split().index('[')+1:l4.strip().split().index(']')])
                dist.append(''.join(map(str,l4.strip().split()[l4.strip().split().index('[')+1:l4.strip().split().index(']')])))
                tmp = sum(label)
                insts[tmp] += 1
                avg += sum(label)

        l4=datafile.readline()
    
    fp.write("Num of instances per label-count (0, 1, 2, ... nlabel)\n")
    for i in range(0, insts.shape[0]):
        fp.write(str(i) + ' ' + str(insts[i])+'\n')
    
    fp.write("Labels frequency: \n")
    
    aux=np.zeros(shape=(labels, 2))
    
    for i in range(0, labels):
        aux [i] = (sum(int(row[i]) for row in dist), i+1)
        
        
    aux = aux[(-aux[:,0]).argsort()]
    
    for s in aux:
        fp.write(str(int(s[1]))+' '+str(int(s[0]))+'\n')

    countr=Counter(dist)
    fp.write ("Label combinations frequency: \n")
    for value, count in countr.most_common():
        fp.write(str(int(value, 2))+' '+ str(count)+'\n')
    #print countr
    un_combs=set(dist)
    #print sorted(un_combs)
    #print ("----------------")
    fp.write ("Cardinality: ")
    card = avg/(instances*1.0)
    fp.write(str(card)+'\n')
    
    fp.write("Density: ")
    fp.write (str(card/(labels*1.0))+'\n')

    fp.write("Distinct: ")
    fp.write(str(len(un_combs))+'\n')
   
    datafile.close()
    fp.close()
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    #insts[] is the vector to plot
    flbs = np.trim_zeros(insts, 'b')
    objects=range(0, flbs.shape[0])
    y_pos = np.arange(len(objects))
    plt.figure(figsize=(15,9))
    plt.bar(y_pos, flbs, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Instances')
    plt.xlabel('Num of active labels')
    plt.title(df+': '+'Label frecuency')
    plt.margins(x=0.01)
    for i,j in zip(flbs, y_pos):
        plt.annotate(str(flbs[j]), xy=(j,i+(np.max(flbs)*0.01)), horizontalalignment='center')

    plt.savefig('./datasets/'+df+'freclbs.png')
    plt.close()

    #Division on python 2.7 returns int by default, 
    #in python3 it returns float so we have to "force" float div on python2.7

def main():
    dataset = {
    'classif20x20week_callin',
    'classif20x20week_callout',
    'classif20x20week_smsin',
    'classif20x20week_smsout',
    'classif20x20week_internet',
   #'Delicious',
   #'bookmarks',
   #'mediamill',
   #'tmc2007',
   #'bibtex',
   #'Corel5k',
    'emotions',
   #'Enron',
   #'genbase',
   #'medical',
   #'scene',
   #'yeast'
    }
    for ds in dataset:
        print ("dataset: "+ ds)
        cardinality(ds)
  
if __name__== "__main__":
    main()
