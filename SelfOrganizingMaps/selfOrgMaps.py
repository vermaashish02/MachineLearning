# -*- coding: utf-8 -*-
import gc
import pandas as pd
from numpy import genfromtxt,array,linalg,zeros,mean,std,apply_along_axis
from __future__ import division
from pylab import plot,axis,show,pcolor,colorbar,bone
from numpy import (array, unravel_index, nditer, linalg, random, subtract,
                   power, exp, pi, zeros, arange, outer, meshgrid, dot)

"""
Author: Ashish Verma
This code is a simple implementation of self organizing maps. This code was developed to demonstrate the basic working of self organizing maps.
So use it with caution and after thorough review, for any production usage.
Feel free to copy/modify/etc. for any purpose.
"""

class selfOrgMaps:
    # dataFrame - input data presented as dataframe
    # gridSize - size of the grid  - [6,6]
    # initLearningRate - Initial learning rate - 0.5
    # initSigma - Initial Sigma value - 2.0
    # noOfIterations - no of examples you want to consider (usually the length of the data)
    # radiusSize - radius size of the neighborhood you want to define - 2
    #inputWeights  - usually kept "" for random initialization but you can provide them here after initializing separately
    def __init__(self, dataFrame, gridSize, initLearningRate, initSigma, noOfIterations, radiusSize, inputWeights):
        #Initializing the SOM grid size
        self.gridSize = gridSize
        #Initialize learning rate
        self.initLearningRate = initLearningRate
        self.dataFrame = dataFrame
        self.random_generator = random.RandomState()
        #-------------Initialize the node weights if node weights are not passed-----------#
        if(inputWeights == ""):
            self.nodeWeights = self.initNodeWeights()
        else:
            self.nodeWeights = inputWeights
        print "nodeWeights", self.nodeWeights
        #----------------------------------------------------------------------------------#
        #Define the no. of iterations (this is usually equal to the length of data)
        self.noOfIterations = noOfIterations
        #Initialize grid by assigning ids to nodes
        self.grid = self.initGrid()
        self.tau = noOfIterations/radiusSize
        self.initSigma = initSigma

    #This function initializes the grid with node ids
    def initGrid(self):
        gridList = []
        for i in range(0, self.gridSize[0]):
            for j in range(0, self.gridSize[1]):
                gridList.append((i,j))
        return np.reshape(np.array(gridList),(self.gridSize[0],self.gridSize[1],2))

    #Initialize the weights of the nodes
    def initNodeWeights(self):
        #Currently weights are randomly initialized but it can be improved by using other ways like PCA etc.
        self.nodeWeights = self.random_generator.rand(self.gridSize[0],self.gridSize[1],self.dataFrame.shape[1])
        #Normalize the weights
        self.nodeWeights = np.array([x/linalg.norm(x) for x in self.nodeWeights])        
        return self.nodeWeights
    
    #Calculate sigma for current iteration
    def calculateSigma(self, currentIteration):
        return self.initSigma*exp(-1*currentIteration/self.tau)
    
    #Identify the best matching unit for the current input
    def identifyBMU(self, inputData):
        distList = []
        distanceFromNodes = (inputData - self.nodeWeights)
        for row in distanceFromNodes:
            distList.append(linalg.norm((row), axis = 1))
        distList = np.array(distList)
        return np.unravel_index(distList.argmin(), distList.shape)
    
    #Update neighborhood parameter for every node based on BMU and sigma
    def updateThetaForGrid(self, inputData, currentIteration):
        indexBMU = self.identifyBMU(inputData)
        print "indexBMU", indexBMU
        sigmaValue = self.calculateSigma(currentIteration)
        distFromBMU = self.grid - indexBMU
        distList = []
        for row in distFromBMU:
            distList.append(linalg.norm((row), axis = 1).tolist())
        #return Θ(t) = exp(-(nodeDistFromBMU)^2 / 2*(σ(t))^2)
        return np.array([exp(-1*(S**2)/(2*(sigmaValue**2))).tolist() for S in np.array(distList)])
    
    #Update weights of all nodes (considering neighborhood criteria)
    def updateNodeWeights(self, inputData, currentIteration):
        thetaGrid = self.updateThetaForGrid(inputData, currentIteration)
        learningRate = self.initLearningRate*exp(-1*currentIteration/self.tau)
        inputToNodeDist = (inputData - self.nodeWeights)
        
        weightLearning = np.array([[learningRate*x*y for (x,y) in zip(i,j)] for (i,j) in zip(thetaGrid, inputToNodeDist)])
        self.nodeWeights = np.array([(i + j) for (i,j) in zip(self.nodeWeights, weightLearning)])
        self.nodeWeights = np.array([[x/linalg.norm(x) for x in y] for y in self.nodeWeights])
    
    #Plot U-Matrix
    def plotUMat(self, iterationNo, saveImageFlag, pathToSaveImage):
        fig = plt.figure(iterationNo)
        uMap = self.uMatrixMap().T
        plt.bone()
        plt.pcolor(uMap,cmap='RdBu') # plotting the distance map as background
        plt.title('Iteration - ' + ("%4d" %iterationNo))
        plt.colorbar()
        if(saveImageFlag == 1):
            fig.savefig(pathToSaveImage + '\UMatrix_Iteration_' + ("%04d" %iterationNo) + '.png')
            fig.clf()            
            plt.close(fig)
            gc.collect()
        return fig
    
    #This is where SOM starts
    #dataFrame - input data presented as dataframe. Normalize the data separately before presenting the data to the code.
    #saveImageFlag - Flag for plotting U-Matrix for every iteration. This can be used for looking at how learning happens over iterations. 
    #But there is a memory leak which may become a bottleneck for higher number of iterations. So beware.
    #pathSavePlots - provide the path where you would like to store the U-Matrix plots
    def startFullSOM(self, dataFrame, saveImageFlag, pathSavePlots):
        for i in range(0, self.noOfIterations):
            inputData = np.array(dataFrame.iloc[np.random.choice(dataFrame.shape[0], 1)])[0]
            self.updateNodeWeights(inputData, i)
            if(saveImageFlag == 1):
                self.plotUMat(i, saveImageFlag, pathSavePlots)
            print "Iteration No - ", i

    #Function to identify the neighborhood
    def identifyNeighborNodes(self, nodeUnderFocus):
        neighborNodeList = []
        for i in xrange(nodeUnderFocus[0] - 1, nodeUnderFocus[0] + 2):
            #print "i", i
            if(i >= 0 and i <= (self.gridSize[0] - 1)):
                for j in xrange(nodeUnderFocus[1] - 1, nodeUnderFocus[1] + 2):
                    #print "j", j
                    if(j >= 0 and j <= (self.gridSize[1] - 1)):                    
                        if (i, j) != (nodeUnderFocus[0], nodeUnderFocus[1]):
                            neighborNodeList.append((i, j))
        return neighborNodeList

    #Prepare U-Matrix for SOM representation
    def uMatrixMap(self):
        uMatrixNodes = zeros((self.nodeWeights.shape[0], self.nodeWeights.shape[1]))        
        for i in range(0, self.nodeWeights.shape[0]):
            for j in range(0, self.nodeWeights.shape[1]):
                nodeUnderFocus = np.array([i,j])
                neighbourNodes = self.identifyNeighborNodes(nodeUnderFocus)
                #print "neighbourNodes", neighbourNodes
                for k in range(0, len(neighbourNodes)):
                    uMatrixNodes[i,j] = uMatrixNodes[i,j] + linalg.norm(self.nodeWeights[neighbourNodes[k]] - self.nodeWeights[i,j])
                    #print "k", k
        #Normalize the map values with the maximum value
        uMatrixNodes = uMatrixNodes/uMatrixNodes.max()
        return uMatrixNodes