'''
    This file will hold logic for n-fold cross validation.

    Methods:
        *generateCrossValidationSets(X, Y, k=5, trainingMethod=None) normal creation of cross validation sets.
        *leaveOneOutCrossValidation(X,Y, trainingMethod=None) has test set of length 1. All others become part of training set.
'''
import numpy as np
import random

def generateCrossValdiationSets(X, Y, k=5, trainingMethod=None):
    '''
        Generate each combination of test and validation sets.
        Providing a training method will automatically run the trainingMethod provided.
        If it is not provided the train/test set combo is added to a folds list and returned.

        Parameters:
            X: The list of parameters
            Y: The list of labels
            k: how many partitions to create
            trainingMethod: a function that should have X,Y as its parameters that will be called every time a partition is created
        Result:
            None if trainingMethod is provided. Else folds is returned which is a list of every combination of train and test sets.
    '''

    #calculate the size of each partition
    size = np.shape(X)[0]
    size /= k

    #holds the final datasets
    folds = []
    #randomize dataset order
    indexOrdering = getRandomOrdering(X)
    #partition
    for testSetPart in range(k):
        index = 0
        currSet = []
        #for each partition
        for i in range(k):
            partX = []
            partY = []
            #for each item that would belong in the first partition
            for j in range(int(size)):
                #add the item
                partX.append(X[indexOrdering[index]])
                partY.append(Y[indexOrdering[index]])
                #move index forward
                index += 1
            #add this to this folds set
            currSet.append((partX, partY))
        #choose the set marked by testSetPart as the partition that will be the test set.
        testX = currSet[testSetPart][0]
        testY = currSet[testSetPart][1]
        trainX = []
        trainY = []
        #combine other sets
        for i in range(k):
            #if pos i is not delegated as the testSet, append its items to the train set
            if i != testSetPart:
                #get the x, y sets at this index
                currX, currY = currSet[i]
                #for each item
                for j in range(len(currX)):
                    #add them to the training set
                    trainX.append(currX[j])
                    trainY.append(currY[j])
        #if a method is provided. this directly runs that trainer. This can save memory in preventing the creation of folds
        if trainingMethod != None:
            trainingMethod((np.array(trainX), np.array(trainY), np.array(testX), np.array(testY)))
        else:
            folds.append((np.array(trainX), np.array(trainY), np.array(testX), np.array(testY)))
    return folds


def leaveOneOutCrossValidation(X,Y, trainingMethod=None):
    '''
        Creates each possibility for leave one out cross validation
        
        Note:
            It does not seem ideal to run this without providing a trainingMethod.

        Parameters:
            X: The list of parameters
            Y: The list of labels
            trainingMethod: a function that should have X,Y as its parameters that will be called every time a partition is created
        Result:
            None if trainingMethod is provided. Else folds is returned which is a list of every combination of train and test sets.

    '''
    
    #get size of input
    size = np.shape(X)[0]
    #get shuffler
    indexOrdering = getRandomOrdering(X)
    #holds the output datasets
    folds = []
    #holds the shuffled data
    shuffledX = []
    shuffledY = []
    #shuffle data
    for i in indexOrdering:
        shuffledX.append(X[i])
        shuffledY.append(Y[i])

    #for each index in the size
    #each index will be a test sample
    for testExample in range(int(size)):
        #define test to be the testExample index
        testX = shuffledX[testExample]
        testY = shuffledY[testExample]
        #define rest to be training
        trainX = np.delete(np.copy(shuffledX), testExample)
        trainY = np.delete(np.copy(shuffledY), testExample)
        if trainingMethod != None:
            trainingMethod((np.array(trainX), np.array(trainY), np.array(testX), np.array(testY)))
        else:
            folds.append((np.array(trainX), np.array(trainY), np.array(testX), np.array(testY)))
    
    return folds


def getRandomOrdering(X):
    '''
        Instead of shuffling an index, this generates a shuffled list of indecies

        Parameters:
            X: List of parameters to learn on
        
        Result:
            indexOrdering: A list of shuffled indexes. same length as X.
    '''
    indexOrdering = []
    size = np.shape(X)[0]
    for i in range(int(size)):
        indexOrdering.append(i)
    random.shuffle(indexOrdering)
    return indexOrdering