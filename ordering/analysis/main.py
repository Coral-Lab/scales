## This file runs the main analysis of ranking data, aggregating the ranks of informants.
## The program takes two arguments, the first is the file where the data from the informants is
## and the second is where to output the aggregated ranks.

## The input file is expected to have one column for each scale, with a name of
## response_<scaleIdentifier>, where scaleIdentifier is a single letter. Each cell
## should have a comma seperated list in it that will be split and handled by the program

## The output has a format of word,scale,tau_k

import cct
import pandas
import sys
import numpy
from sklearn import preprocessing

data = pandas.read_csv(sys.argv[1],quoting=3,delimiter="\t")

#Calculate competency scores
competencies, eigenRatios, originalCompetencies = cct.processData(data)
responseMatrices = cct.buildMatrices(data)

sortedWords = []

#For each scale in the dataset
for dataset, responseMatrix in responseMatrices.items():
    datasetLetter = dataset.split('_')[1]
    estimatedRank = {}

    # Transform the response matrix into a scaled matrix
    # Also flip the rankings of the informants that orginially had negative competencies
    responseMatrix = pandas.DataFrame(preprocessing.scale(cct.reverseNegatives(responseMatrix,originalCompetencies[dataset]),axis=1), columns = responseMatrix.columns)
    local_comp = competencies[dataset]

    #Calculate tau_k for each word in the scale
    for col in responseMatrix:
        estimatedRank[col] = local_comp.dot(responseMatrix[col])
    sortedWords.append(pandas.DataFrame({'scale': numpy.repeat(datasetLetter,len(estimatedRank)),'tau_k':pandas.Series(estimatedRank)}).reset_index())

pandas.concat(sortedWords,ignore_index=True).sort_values(['scale','tau_k']).to_csv(sys.argv[2],index=False,header=False)

print pandas.Series(eigenRatios)
