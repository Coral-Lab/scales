## This file runs the main analysis to determine which words
## are members of a scale. It requires two command line
## arguments, the location of the file with the informant's
## responses, and a location to output the analysis to.

## The input file is expected to two columns for each scale
## data was elicted for. The columns should be named
## Answer.prompt_<scaleIdentifier> and Answer.data_<scaleIdentifier>,
## where scale identifier is a single letter.

## The output file is formated as a csv with columns scale,word,G_k

import cleaning
import cct
import sys
import pandas
import numpy
from collections import defaultdict

# Function to return G_k as defined in Batchelder, William H., and A. Kimball Romney. 1988.
# “Test Theory without an Answer Key.” Psychometrika 53 (1): 71–92.
# The parameters are the competency vector D, the resposne vector X,
# and a bias variable g, which is cacluated per scale
def estimator(D,X,g):
    a = 1 - D
    ratio1 = ((D + g * a) * ( 1 - g * a))/(a ** 2 * g * (1 - g))
    ratio2 = (1 - g * a)/(a * (1 - g))
    return X * numpy.log(ratio1) - numpy.log(ratio2)

# Perform spellchecking and lowercasing on data
responses, prompts = cleaning.processFile(sys.argv[1])

# Perform CCT on data and cap competencies that are great than 1
# at .999
cctResults = cct.processData(responses,prompts)
competencyMatrix = pandas.DataFrame(cctResults[0])
competencyMatrix[competencyMatrix > 1] = .999
print pandas.Series(cctResults[1])

responseMatrices = cct.buildMatrices(responses)
correctSet = {}

correct = defaultdict(lambda:defaultdict())
# Calculate G_k for each word in each scale
for dataset, responseMatrix in responseMatrices.items():
    datasetLetter = dataset.split('_')[1]
    local_comp = competencyMatrix[dataset][responseMatrix.index]

    # Get the average number of words per informant for each scale
    # and the total number of words given per scale. This is used to derive the bias variable

    avgResponse =  responseMatrix[responseMatrix==1].count(1).mean()
    length = responseMatrix.shape[1]

    prompt = prompts['Answer.prompt_' + datasetLetter]

    for col in responseMatrix:
        #Exclude responses for which the informant was given a word as a prompt word
        contains = prompt.apply(lambda x: col in x)
        good = prompt[contains == False].index
        correct[datasetLetter][col] =(estimator(local_comp.ix[good],responseMatrix[col].ix[good],avgResponse/length)).sum()

results = pandas.DataFrame(correct).stack().reset_index().rename(columns={"level_0":"word","level_1":"scale",0:"G_k"})
results[["scale","word","G_k"]].sort_values(["scale","G_k"],ascending=[True,False]).to_csv(sys.argv[2],index=False,header=False)
