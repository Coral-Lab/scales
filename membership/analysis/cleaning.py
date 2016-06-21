## This file contains the methods necessary to
## preprocess the elicitation data. This ensures that
## minor spelling variations as well as all differences
## in csae are accounted form

import pandas
import hunspell
import re
from collections import Counter

speller = hunspell.HunSpell(
    '/usr/share/hunspell/en_US.dic', '/usr/share/hunspell/en_US.aff')

# Lower case the data
def homongonizeAnswer(dataElement):
  if type(dataElement) == list:
    return [x.lower().strip() for x in dataElement if x.strip() != '']
  elif type(dataElement) == str:
    return dataElement.lower().strip()
  else:
    return dataElement

# Fix the spelling if hunspell reports that the word is mispelled.
# Always accept the first suggestion if it is mispelled.
def fixSpelling(dataElement):
    #print dataElement
    if type(dataElement) == list:
      return [speller.suggest(x)[0] if not speller.spell(x) and ' ' not in x and len(speller.suggest(x)) > 0 and speller.suggest(x)[0].lower() != x else x for x in dataElement]


# Split the informants response into a python list
def splitAnswers(dataFrame):
    answers = dataFrame.filter(regex='data_[a-z]$')
    splits = answers.applymap(lambda x: x.split(','))
    #dataFrame.loc[:, answers.columns.tolist()] = splits
    #return dataFrame
    return splits

# Split the prompt words given into a python list.
def splitPrompts(dataFrame):
    prompts = dataFrame.filter(regex="Answer.prompt_[a-z]")
    prompts = prompts.apply(lambda x: x.str.replace('and','').str.split(","))
    prompts = prompts.applymap(lambda x: map(lambda y: y.strip(),x))
    return prompts

# Perform all the preprocessing in one method
def processFile(mturk_results):
    data = pandas.read_csv(mturk_results)
    datasplit = splitAnswers(data.dropna(axis=1,how='all'))
    dataprompts = splitPrompts(data)
    dataclean = datasplit.applymap(homongonizeAnswer).applymap(fixSpelling).applymap(homongonizeAnswer)

    return dataclean, dataprompts
