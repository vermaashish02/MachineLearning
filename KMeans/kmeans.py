"""
Author: Ashish Verma
This code was developed to give a clear understanding of what goes behind the curtains in K-Means clustering (hard and soft).
Feel free to use/modify/improve/etc. but this may not be efficient code for production related usage (especially where data is large) 
so thoroughly review and test the code before usage.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class kMeansClustering:
    def __init__(self, dataFrame, typeOfMeanInitialization, noOfClusters):
        #Initialize means randomly
        if(typeOfMeanInitialization == 'random'):
            randomArray = np.random.choice(len(dataFrame), noOfClusters, replace=False)
            self.meanOfClusters = np.array(dataFrame)[randomArray,:]
        else:
            #Initialize means using farthest first point
            if(typeOfMeanInitialization == 'ffp'):
                self.meanOfClusters = self.ffpInit(np.array(dataFrame), noOfClusters)
        #Commenting below line as it was meant for scatter plots saved by plotScatterWithMeans function (which is not a part of this class)
        #self.objPlot = plt.figure(1)
    #This function finds out farthest first points
    def ffpInit(self, dataArray, noOfClusters):
        #List containing all first means
        meansList = []
        #Find all FFP points
        for iCluster in range(0, noOfClusters):
            if(iCluster == 0):
                #Initialize first mean randomly
                firstMean = dataArray[np.random.choice(dataArray.shape[0], 1)][0].tolist()
                print "firstMean", firstMean
                meansList.append(firstMean)
            else:
                distanceList = []
                #Find the distance of all points from the means initialized so far
                distanceList = self.chunkedDistanceCalculation(dataArray, meansList, 20000)
                #Find the minimum of the distances from all means
                minDistanceList = [np.min(x) for x in distanceList]
                #Find the point which has maximum of minimum distance from the means
                newMeanIndex = minDistanceList.index(max(minDistanceList))
                #Extract the actual values
                newMean = dataArray[newMeanIndex][:].tolist()
                #Store the mean in the list
                meansList.append(newMean)
        return  meansList
        
    #If the number of datapoints are very high and hardware is not that great then this function helps in finding the distance
    #in chunks (defined by chunkSize). So compromise on performance but still better than iterating over data one by one.
    def chunkedDistanceCalculation(self, dataFrame, inputMeanList, chunkSize):
        #Just to conver up a situation where we define the chunkSize greater than no. of rows
        if(chunkSize > len(dataFrame)):
            chunkSize = len(dataFrame)
        #array of means provided as input
        meanArray = np.array(inputMeanList)
        #Define a distance array which has rows same as dataset and columns as no. of means
        distanceArray = np.zeros((len(dataFrame), meanArray.shape[0]))
        #Quotient
        quot = divmod(len(distanceArray), chunkSize)[0]
        #Remainder
        remainder = divmod(len(distanceArray), chunkSize)[1]
        #Adjusting remainder in no. of loops (incase the chunkSize is smaller than total lenght of data
        if(remainder == 0):
            loops = quot
        else:
            loops = quot + 1
        #Iterate over all means
        for i in range(0, meanArray.shape[0]):
            print "iterating over means", i
            startRow = 0
            jumps = chunkSize
            #Calculate distance based on chunkSize
            for iRow in range(0, loops):
                if(iRow == (loops - 1)):
                    tempData = dataFrame[startRow:, :]
                    distanceArray[startRow : ,i:i+1] = np.array([np.diag(np.dot((tempData-meanArray[i]),(tempData-meanArray[i]).T)).tolist()]).T
                else:
                    endRow = startRow + jumps
                    tempData = dataFrame[startRow : endRow, :]
                    distanceArray[startRow : endRow,i:i+1] = np.array([np.diag(np.dot((tempData-meanArray[i]),(tempData-meanArray[i]).T)).tolist()]).T
                    startRow = endRow
        #Store the distance from arrays as a list to return
        distanceList = distanceArray.tolist()
        return distanceList
    
    #Calculate means for both hard as well as soft k-means clustering
    def calculateMeans(self, dataFrame, noOfClusters, hardOrSoft, oldMeanList):
        fullMeanList = []
        #Remove columns which contain cluster nos. (for hard) or cluster fraction (for soft)
        features = [col for col in dataFrame.columns.values if 'clusterNo' not in col]
        tempData = dataFrame[features]
        if(hardOrSoft == 'hard'):
            clustersAssigned = np.sort(dataFrame.clusterNo.unique())
            #Calculate means of all clusters
            for iCluster in range(0, len(clustersAssigned)):
                #Hard clustering means
                #if(hardOrSoft == 'hard'):
                    #Calculate cluster mean
                clusterMean = np.array(np.mean(tempData[dataFrame['clusterNo'] == clustersAssigned[iCluster]], axis = 0))
                fullMeanList.append(clusterMean)                
        else:
            clustersList = [clustNo for clustNo in dataFrame.columns.values if 'clusterNo' in clustNo]
            print "clustersList", clustersList
            #Calculate means of all clusters
            for iCluster in range(0, len(clustersList)):            
                #Calculate cluster means for soft clustering
                dataFractionArray = np.array(dataFrame['clusterNo' + str(iCluster)])
                clusterMean = sum(np.array([x * y for (x,y) in zip(np.array(tempData), dataFractionArray)]),axis = 0)/sum(dataFractionArray)
                fullMeanList.append(clusterMean)
        #--------------Handling a scenario where no data point is assigned to a cluster (Hard)--------------#
        if(len(fullMeanList) != noOfClusters):
            print "finding and inserting the isolated mean....................."
            print "partial means list", fullMeanList
            fullNewMeansList = fullMeanList[:]
            #Extract the cluster no which is missing (if any)
            isolatedClusters = list(set(range(0,noOfClusters)) - set(clustersAssigned))
            if(len(isolatedClusters) > 0):
                for j in range(0, len(isolatedClusters)):
                    fullNewMeansList.insert(isolatedClusters[j], oldMeanList[isolatedClusters[j]])                
            print "full means list", fullNewMeansList
            fullMeanList = fullNewMeansList
        #-------------------------------------------------------------------------------------------------#
        print "returning means list", fullMeanList
        #Return cluster means
        return fullMeanList
                
    
    #This is the main K-Means clustering function
    #dataFrame - input data in form of dataframe
    #noOfClusters - total number of clusters
    #hardOrSoft - 'hard' for hard clustering and 'soft' for soft clustering
    #maxLabelsThreshold - threshold for label changes in one iteration below which the code will stop conerging
    #maxIterationsThreshold - threshold for total no. of iterations, above which the code will stop converging
    def runKMeansClustering(self, dataFrame, noOfClusters, hardOrSoft, maxLabelsThreshold, maxIterationsThreshold):
        #Assign initialized means as new cluster means
        newMeanOfClusters = self.meanOfClusters
        #Initialize old means. This will be used for storing means from previous iteration
        oldMeanOfClusters = np.zeros(np.array(newMeanOfClusters).shape)
        #This list will store label changes from each iteration (as compared to previous iteration)
        labelChangingList = []
        #Initialize beta for soft clustering
        beta = 1.1
        #Initialize iteration count
        iteration = 0
        #Loop untill the convergence happens (i.e. cluster means do not change in next iteration)
        while (not np.array_equal(oldMeanOfClusters, newMeanOfClusters)):
            #filter only data features
            featureColumns = [col for col in dataFrame.columns if 'clusterNo' not in col]
            tempData = dataFrame[featureColumns]
            #Calculate the distance of all data point from all means
            distanceList = self.chunkedDistanceCalculation(np.array(tempData), newMeanOfClusters, 20000)
            #Initialize no. of labels changed
            noOfLabelsChanged = 0
            #Calculate cluster numbers for all data points (in hard clustering)
            if(hardOrSoft == 'hard'):
                clusterNoArray = [x.index(min(x)) for x in distanceList]
                #Store cluster numbers in the dataframe as a separate column
                dataFrame['clusterNo'] = clusterNoArray
                #Calculate labels changed only from iteration = 1 (2nd iteration) as in 1st iteration (iteration = 0) there will be no label changes
                if(iteration >=1):
                    noOfLabelsChanged = len([clusterOld for clusterOld, clusterNew in zip(oldClusterList, clusterNoArray) if clusterOld != clusterNew])
                #Store new cluster numbers as old cluster nos. to be used for calculation of next iteration
                oldClusterList = clusterNoArray
            #Calculate data fractions for each data point assigned for each mean
            else:
                #Normalize distances as exponential of large distances was creating an issue on my side.
                distanceList = [x/linalg.norm(x) for x in distanceList]
                #Calculate exponential term for soft clustering
                dataExpArray = exp(-1*beta*np.array(distanceList))
                #Calculate sum of expoenential terms across means direction to normalize in next step
                sumFractionsArray = np.array(sum(dataExpArray, axis = 1))
                #Normalize exponential terms
                dataFractionArray = [x/y for (x,y) in zip(dataExpArray, sumFractionsArray)]
                for iClusterNo in range(0, np.array(dataFractionArray).shape[1]):
                    #Store fraction values in respective cluster columns
                    dataFrame['clusterNo' + str(iClusterNo)] = np.array(dataFractionArray)[:, iClusterNo]
                    if(iteration >=1):
                        #Calculate labels changed from 2nd iteration onwards (iteration = 1)
                        noOfLabelsChanged = noOfLabelsChanged + sum(abs(np.array(dataFractionArray)[:, iClusterNo] - oldClusterList[:,iClusterNo]))/2
                #Store fractions calculate as old fraction list for usage in next iteration
                oldClusterList = np.array(dataFractionArray)
                #increment beta value
                beta = beta + 0.1
            #Define dictionary to store labels changed over one iteration related information
            labelChangingDict = {}
            print "iteration", iteration
            #Store means as old means
            oldMeanOfClusters = newMeanOfClusters
            #Calculate new means of new clusters
            newMeanOfClusters = self.calculateMeans(dataFrame, noOfClusters, hardOrSoft, oldMeanOfClusters)
            #----------------scatter plot with means (plotScatterWithMeans function is not a part of 'kMeansClustering' class---------------------#
            #----------------so I have commented it out but it is just a plain  matplotlib.pyplot based plotting----------------------------------#
            #plotScatterWithMeans(dataFrame, newMeanOfClusters, 'hard', (iteration + 1), self.objPlot)
            #-------------------------------------------------------------------------------------------------------------------------------------#
            #Store information only from 2nd iteration (iteration = 1)
            if(iteration>=1):
                labelChangingDict['iterationNo'] = iteration
                labelChangingDict['labelsChanged'] = noOfLabelsChanged
                #Store data from every iteration into a list which can be used later to see convergence properties
                labelChangingList.append(labelChangingDict)                
                #Stop converging if the no. of labels changed in one iteration is less than or equals to the threshold defined
                if(noOfLabelsChanged <= maxLabelsThreshold):
                    print "threshold defined by maxLabelsThreshold reached, so breaking !"
                    break
            iteration = iteration + 1
            #Stop converging if the total no. of iterations exceeds the threshold defined
            if(iteration > maxIterationsThreshold):
                print "maximum no. of iterations allowed exceeded, so breaking !"
                break
        #Store the final cluster means within object
        self.meanOfClusters = newMeanOfClusters
        return dataFrame, labelChangingList