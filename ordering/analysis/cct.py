## This file contains the methods needed to perform the factor anaylsis
## to derive the competencies of the informants.

import pandas
from sklearn import preprocessing
import numpy
import rpy2.robjects.packages as rpackages
import rpy2.robjects.numpy2ri as np2ri
import rpy2.robjects.pandas2ri as pandas2ri
import rpy2.robjects as ro
np2ri.activate()
psych = rpackages.importr('psych')

# Convert the string response of each informant into a matrix
# with a column for each word, where each cell[i,j] holds
# informant i's rank of word j.
def buildMatrix(responses):
    results = {}
    for i,l in responses.str.split(",").iteritems():
        row = {}
        for pos,j in enumerate(l):
            row[j] = pos + 1
        results[i] = row
    matrix =  pandas.DataFrame(results)
    matrix = matrix.T
    return matrix.reset_index(drop=True)

# Build a response matrix for each scale, excluding informants who did not
# perform the task for that scale
def buildMatrices(dataFrame):
    matrices = {}
    filtered = dataFrame.filter(regex='\.response_\w$')
    for name in filtered.columns:
        matrices[name] = buildMatrix(dataFrame[name][dataFrame[name] != "{}"])
    return matrices

# Calculate the pairwise correlation between informants as defined in
# Romney, A. Kimball, William H. Batchelder, and Susan C. Weller. 1987.
# "Recent Applications of Cultural Consensus Theory."
# American Behavioral Scientist 31(2): 163-77.
def dotCorr(matrix):
    df = matrix
    M = df.shape[1]
    K = df.shape[0]
    result = numpy.empty((K, K), dtype=numpy.float64)
    for xi,A in df.iterrows():
        Bdf = df.head(xi +1)
        A_s = preprocessing.scale(A.astype(numpy.float64))
        for yi, B in Bdf.iterrows():
            B_s = preprocessing.scale(B.astype(numpy.float64))
            result[xi][yi] = 1.0/M * A_s.dot(B_s)
            result[yi][xi] = 1.0/M * A_s.dot(B_s)
    return result

# Perform factor analysis. Take as input the reponse matrix and then calculates the
# pairwise correlation matrix in this function
def CCT(matrix):
    agreementMatrix = numpy.nan_to_num(dotCorr(matrix))
    numpy.fill_diagonal(agreementMatrix,1)
    factors = psych.fa(agreementMatrix,fm='pa')
    return factors

# Reverse the ranking order for informants whose
# competency is negative
def reverseNegatives(matrix,competencies):
    for i, row in matrix.iterrows():
        if competencies[i] < 0:
            matrix.ix[i] = ((row - max(row)) * -1.0) + 1
    return matrix

# Process the response file from MTURK,
# calculating the competencies and eigenvalue ratios
# for each scale.
def processData(dataFrame):
    originalCompetencies = {}
    competencies = {}
    eigenRatios = {}
    matrices = buildMatrices(dataFrame)

    for dataSet, matrix in matrices.items():
        factors = CCT(matrix)

        results = np2ri.ri2py((factors.rx('loadings')[0])).flatten()
        originalCompetencies[dataSet] = results

        matrix = reverseNegatives(matrix,results)

        factors = CCT(matrix)
        results = np2ri.ri2py((factors.rx('loadings')[0])).flatten()
        competencies[dataSet] = results

        eigen = np2ri.ri2py((factors.rx('values')[0])).flatten()
        eigenRatios[dataSet] = eigen

    return (competencies, eigenRatios, originalCompetencies)
