#!/usr/bin/python3.8
#Converts the .arff dataset into a clearer version to execute some measures like Card, dens and distinct.

import sys
import numpy as np
import arff


# Call
if len(sys.argv) <= 1:
    print ("multilabel-convert.py input-file [output-file-prefix]")
    sys.exit()

# Read arff file
if sys.argv[1].lower().endswith('.arff') == False :
    sys.exit("Dataset format unknown, please use .arff datasets")    

dataset = arff.load(open(sys.argv[1], 'r'))
data = np.array(dataset['data'])

#We have to get the number of clases from the raw file
file = open(sys.argv[1], "r")
line = file.readline()

flag = False
for i in line.split():
    if flag is True:
        number = i
        break
    if (i == "-C") or (i == "-c"):
        flag = True

if (flag==False):
    file.close()
    sys.exit("Wrong format for the dataset header")

if number[-1:] == "'":
    number = number[:-1]
file.close()
#Now we have the number stored, knowing that positive means the first attributes and negative the last ones

nominalIndexArray = []
nominals = []
aux = 0
#from attributes we can get if its nominal
if int(number) > 0:
    for x in dataset['attributes'][int(number):]:
        if (len(x[1]) > 2) and (x[1] != ("NUMERIC" or "REAL" or "INTEGER" or "STRING")):
            nominalIndexArray.append(aux)
            nominals.append(x[1])
        aux +=1
else:
    for x in dataset['attributes'][:int(number)]:
        if (len(x[1]) > 2) and (x[1] != ("NUMERIC" or "REAL" or "INTEGER" or "STRING")):
            nominalIndexArray.append(aux)
            nominals.append(x[1])
        aux +=1

#Split the data in X and Y
if(int(number)>0):
    y = data[:,0:int(number)].astype(int)
    x = data[:,int(number):]
else:
    y = data[:,int(number):].astype(int)
    x = data[:,:int(number)]

if len(nominalIndexArray) > 0:
    #Change the nominal attributes to numeric ones
    index = 0
    X = []
    for k in x:
        numericVector = []
        for i in range(0, len(nominalIndexArray)):
            #Ahora tenemos que crear el vector que le vamos a poner al final de cada 
            checkIfMissing = False
            for aux in nominals[i]:
                if aux == k[nominalIndexArray[i]]:
                    #Add 1 to the array
                    checkIfMissing = True
                    numericVector.append(1)
                else:
                    #Add 0 to the array
                    checkIfMissing = True
                    numericVector.append(0)
            if checkIfMissing is False:
                #Add another 1 to the array
                numericVector.append(1)
            else:
                numericVector.append(0)
        auxVector = np.append(k, [numericVector])
        #Substract that nominals values
        auxVector = np.delete(auxVector, nominalIndexArray)
        X.append(auxVector)
                   
    X = np.array(X)
else:
    X = np.array(x)

# Sparse or dense?
sizeofdouble = 8
sizeofint = 4
sizeofptr = 8
dense_size = len(X)*len(X[0])*sizeofdouble+len(X)*sizeofptr
#nz = np.count_nonzero(X)
#Count_nonzero is not working so i'll count the non zeroes by myself (This will take a little longer but better than not counting them)
nz = 0
for i in range(0, len(X)):
    for j in range(0, len(X[0])):
        if X[i][j] != '0.0':
            nz += 1
sparse_size = nz*(sizeofdouble+sizeofint)+2*len(X)*sizeofptr+len(X)*sizeofint

sparse = False if sparse_size >= dense_size else True

# Use input file as output suffix if no other given
suffix = sys.argv[3] if len(sys.argv) == 4 else sys.argv[1][:sys.argv[1].rfind('.')]

#Complete file

fp = open(suffix+'.complete', 'w')

#Save header
if sparse:
    fp.write('[MULTILABEL, SPARSE]\n')
else:
    fp.write('[MULTILABEL, DENSE]\n')
fp.write('$ %d\n' % len(X)) #Number of objects
fp.write('$ %d\n' % len(X[0])) #Number of attributes
fp.write('$ %d\n' % abs(int(number))) #Number of labels

#Data
for i in range(0, len(X)):
	if sparse:
		for j in range(0, len(X[i])):
			if(X[i][j] != '0.0'):
				fp.write(str(j+1)+':'+str(X[i][j])+' ')
			if(X[i][j] == 'YES'):
				fp.write('1'+' ')		
	else:
		for j in range(0, len(X[i])):
			if(X[i][j] == 'YES'):
				fp.write('1'+' ')
			elif (X[i][j] == 'NO'):
				fp.write('0'+' ')
			else:
				fp.write(str(X[i][j])+' ')
    
	fp.write('[ ')
	for j in range(0, len(y[i])):
		if y[i][j] == '0.0':
			aux = str(y[i][j]).split('.')[0]
			fp.write(str(int(aux))+' ')
		else:
			fp.write(str(int(y[i][j]))+' ')
	fp.write(']\n')

#Save header
fp.close()
