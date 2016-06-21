## This file contains the methods needed to perform the factor anaylsis
## to derive the competencies of the informants.

import sys
from scipy.stats import pearsonr
import pandas
import numpy as np
import random
import numpy
import rpy2.robjects.packages as rpackages
import rpy2.robjects.numpy2ri as np2ri
import rpy2.robjects as ro
np2ri.activate()
psych = rpackages.importr('psych')

# Calculate the competency vectors and
# eigen value ratios for a set of scales and prompts
def processData(data,prompts):
    expandedMatrix = buildMatrices(data)
    competencies = {}
    eigenRatios = {}
    meanCompetencies = {}
    negativeCompetencies = {}
    for dataSet, matrix in expandedMatrix.items():
        sys.err.println("Processing " + dataSet)

        prompt = prompts['Answer.prompt_' + dataSet.split('_')[1]]

        # Calculate competencies for this scale
        factors = CCT(matrix, prompt)
        competencies[dataSet] = np2ri.ri2py((factors.rx('loadings')[0])).flatten()
        eigen = np2ri.ri2py((factors.rx('values')[0])).flatten()

        eigenRatios[dataSet] = eigen[0]/eigen[1]
        meanCompetencies[dataSet] = competencies[dataSet].mean()
        negativeCompetencies[dataSet] = (competencies[dataSet] < 0).sum() * 1.0/len(competencies[dataSet])

    return (competencies,eigenRatios,meanCompetencies,negativeCompetencies)

# Convert the resposnes of each informant into a binary
# response matrix, where a 1 in cell [i,j] indicates
# informant i listed word j.
def buildMatrix(responses):
    matrixDict = {}
    for i,l in responses.iteritems():
        row = {}
        for pos,j in enumerate(l):
            row[j] = 1
        length = len(row)
        matrixDict[i] = row
    matrix =  pandas.DataFrame(matrixDict)
    return matrix.fillna(0).T

# Build the resposne matrices for each scale that was elicited
def buildMatrices(dataFrame):
    matrices = {}
    filtered = dataFrame.filter(regex='\.data_\w$')
    for name in filtered.columns:
        matrices[name] = buildMatrix(dataFrame[name])
    return matrices

# Calculate the informant by informant correlation matrix and
# perform factor anaylsis to estimate the competencies of the
# informants
def CCT(matrix,prompt):
    agreementMatrix = modifiedCorr(matrix,prompt)
    numpy.fill_diagonal(agreementMatrix,1)
    factors = psych.fa(agreementMatrix,fm='pa')
    return factors

# Calculate the informant by informant correlation matrix
# using the method discussed in the paper. This method
# seeks a compremise between giving credit to informants who
# didn't see a word, while not punishing them if the informant
# they are being compared against did list that word
def modifiedCorr(matrix,prompts):

    df = matrix.copy()
    promptDF = prompts.copy()
    K = df.shape[0]
    result = np.empty((K, K), dtype=np.float64)

    for xi,A in df.iterrows():
        Bdf = df.head(xi +1)
        prompt_A = promptDF.iat[xi]
        A_index = A.index
        for yi, B in Bdf.iterrows():# range(xi+1):
            A_tmp = A.copy()
            B_tmp = B.copy()
            prompt_B = promptDF.iat[yi]

            # Series of tests to determin if either of informant A or B's prompt words were listed
            # by the other

            M1 = A.at[prompt_B[0]] if prompt_B[0] in A_index and prompt_B[0] not in prompt_A else 0
            M2 = A.at[prompt_B[1]] if prompt_B[1] in A_index and prompt_B[1] not in prompt_A else 0
            M3 = A.at[prompt_B[2]] if prompt_B[2] in A_index and prompt_B[2] not in prompt_A else 0

            N1 = B.at[prompt_A[0]] if prompt_A[0] in A_index and prompt_A[0] not in prompt_B else 0
            N2 = B.at[prompt_A[1]] if prompt_A[1] in A_index and prompt_A[1] not in prompt_B else 0
            N3 = B.at[prompt_A[2]] if prompt_A[2] in A_index and prompt_A[2] not in prompt_B else 0

            # If a A listed one of B's prompt words in their response, give B credit for
            # the purposes of this correlation only
            if M1 == 1:
                B_tmp.at[prompt_B[0]] = 1
            if M2 == 1:
                B_tmp.at[prompt_B[1]] = 1
            if M3 == 1:
                B_tmp.at[prompt_B[2]] = 1


            # If a B listed one of A's prompt words in their response, give A credit for
            # the purposes of this correlation only
            if N1 == 1:
                A_tmp.at[prompt_A[0]] = 1
            if N2 == 1:
                A_tmp.at[prompt_A[1]] = 1
            if N3 == 1:
                A_tmp.at[prompt_A[2]] = 1

            r = pearsonr(A_tmp,B_tmp)[0]

            result[xi][yi] = r
            result[yi][xi] = r
    return result
