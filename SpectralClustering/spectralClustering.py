"""
Author: Ashish Verma
This code was developed to give a clear understanding of what goes behind the curtains in Spectral clustering.
Feel free to use/modify/improve/etc. 

Caution: This may not be an efficient code for production related usage (especially where data is large) so thoroughly review and test the code before any usage.
"""

from sklearn.neighbors import NearestNeighbors

class spectralClustering:
    def __Init__(self):
        #variable to store Laplacian matrix
        self.laplacianMatrix = ""
        #variable to store Degree matrix        
        self.degreeMatrix = ""
        #variable to store Adjacency matrix        
        self.adjacencyMatrix = ""

    #'constructAdjacencyMatrix' function calculates adjacency based on k-nearest neighbors
    #dataFrame - pandas dataframe
    #k_neighbors - number of nearest points to be considered as neighbors
    #This function uses sklearn.neighbors.NearestNeighbors() function for K-nearest neighbors
    def constructAdjacencyMatrix(self, dataFrame, k_neighbors):
        model = NearestNeighbors(n_neighbors = k_neighbors, algorithm='ball_tree').fit(dataFrame.values)
        distanceArray, indicesArray = model.kneighbors(dataFrame.values)
        #Store the affinity
        affinity = model.kneighbors_graph(dataFrame.values)
        self.adjacencyMatrix = affinity.toarray()
    
    #Calculate the degree matrix
    def constructDegreeMatrix(self):
        self.degreeMatrix = np.diag([sum(row) for row in self.adjacencyMatrix])
    
    #Calculate the laplacian from degree matrix and adjacency matrix
    def constructLaplacianMatrix(self):
        self.laplacianMatrix = self.degreeMatrix - self.adjacencyMatrix
    
    #This function executes the spectral clustering
    def runSpectralClustering(self, dataFrame, noOfClusters, k_neighbors):
        #Calculate adjacency matrix
        self.constructAdjacencyMatrix(dataFrame, k_neighbors)
        #Calculate degree matrix
        self.constructDegreeMatrix()
        #Calculate laplacian matrix
        self.constructLaplacianMatrix()
        #Calculate eigenvalues and eigenvectors for laplacian matrix
        eigenValues, eigenVectors = np.linalg.eig(self.laplacianMatrix)
        sortedEigenValueIndex =  np.argsort(eigenValues.real)
        sortedEigenVectors = eigenVectors[:,sortedEigenValueIndex].real
        #Extract only first k eigenvectors
        sortedEigenVectors = sortedEigenVectors[:,:noOfClusters]
        eigenDataFrame = pd.DataFrame(sortedEigenVectors)
        columnNames = eigenDataFrame.columns.values
        columnNames = [str(x) for x in columnNames]
        eigenDataFrame.columns = columnNames
        
        #Do k-means clustering of the first k-eigenvectors
        #Here I have used 'kMeansClustering' class which is a part of my earlier post - https://whyml.wordpress.com/2016/06/01/k-means-clustering-hard/
        #eigenDataFrame - first k eigenvectors passed as pandas dataframe
        #'ffp' - type of initialization (farthest first points)
        #noOfClusters - no of clusters to be identified
        objKMeans = kMeansClustering(eigenDataFrame , 'ffp', noOfClusters)
        #Do hard clustering
        data, labelData = objKMeans.runKMeansClustering(eigenDataFrame , noOfClusters, 'hard', 1, 50)
        #Return the array of cluster labels
        return np.array(data['clusterNo'])