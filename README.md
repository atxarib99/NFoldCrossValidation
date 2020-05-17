# nfoldcross

This class is for creating N-Fold-Cross Validation sets. It is used to test the performance of a model.

The idea is that you have one set of data:

[data]

We then split this up into partitions. For example the default size is 5. It has been proven that the best partioning sizes are 5 and 10.

[test][train][train][train][train]

[train][test][train][train][train]

...

Each of these is then used to train and test the trainer. The results can be used to see how well the trainer is able to learn on different training sets.

You can also perform leave on out cross validation which has a test set of length 1, while the rest become training examples.

## Example

This is the most basic way of using this.

'''python
#import first
import nfoldcross as nfcv
#get folds from nfcv. Provide X (Parameters) and Y (Labels)
folds = nfcv.generateCrossValidationSets(X, Y)
#for each fold
for fold in folds:
    #get partitioned
    trainX, trainY, testX, testY = fold
    #your train method here
    train(trainX, trainY)
    #your accuracy method here
    accuracy(testX, testY)
'''

For memory improvements, a training method can be provided that will run after each fold is created. Then the fold will be destroyed.

'''python
#import first
import nfoldcross as nfcv
#get folds from nfcv. Provide X (Parameters) and Y (Labels) and your training method. Your training method should have parameters for trainX, trainX, trainY, testX, testY. Your training method should also be calculating and ouputting accuracy.
folds = nfcv.generateCrossValidationSets(X, Y, trainingMethod=train)
'''

To use Leave One Out Cross Validation. You can also provide a trainingMethod like above here. Providing a trainingMethod is highly reccomended since all folds will be of size len(X) each of which contains a copy of training and test sets.

'''python
#import first
import nfoldcross as nfcv
folds = nfcv.leaveOneOutCrossValidation(X, Y)
#for each fold
for fold in folds:
    #get partitioned
    trainX, trainY, testX, testY = fold
    #your train method here
    train(trainX, trainY)
    #your accuracy method here
    accuracy(testX, testY)
'''

## Documentation

All documentation has been provided in docstrings. Any merge requests should have docstrings included.
