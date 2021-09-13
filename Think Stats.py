# region Packages
import pandas
import numpy as np
import scipy
import statsmodels
import matplotlib


# endregion Packages

# region Link
# Book
# https://greenteapress.com/thinkstats2/html/index.html

# Souce codes
# https://github.com/AllenDowney/ThinkStats2

# Codebook
# http://www.cdc.gov/nchs/nsfg/nsfg_cycle6.htm
# https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NSFG/Cycle6Codebook-Pregnancy.pdf - Pregnancy Codebook
# https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NSFG/Cycle6Codebook-Female.pdf - Respondent Codebook
# https://www.cdc.gov/nchs/data/nsfg/App2_RecodeSpecs.pdf - Recode Specs


# Run Jupyter notebook
# https://jupyter.readthedocs.io/en/latest/running.html#how-do-i-open-a-specific-notebook
# run all commands below in cmd
# Install
#   pip3 install jupyter

# Steps to open notebook
#   1. navigate to relevent folder
#       cd c:\...\code
#   2. run jupter notebook
#       jupter notebook chap01ex.ipynb


# endregion Link

# region Chapter 1  Exploratory data analysis
# region anecdotal evidence usually fails, because:
# anecdotal evidence usually fails, because:
#     Small number of observations:
#       If pregnancy length is longer for first babies, the difference is probably small compared to natural variation.
#       In that case, we might have to compare a large number of pregnancies to be sure that a difference exists.
#     Selection bias:
#       People who join a discussion of this question might be interested because their first babies were late.
#       In that case the process of selecting data would bias the results.
#     Confirmation bias:
#       People who believe the claim might be more likely to contribute examples that confirm it.
#       People who doubt the claim are more likely to cite counterexamples.
#     Inaccuracy:
#       Anecdotes are often personal stories, and often misremembered, misrepresented, repeated inaccurately, etc.


# endregion anecdotal evidence usually fails, because:

# region 1.1  A statistical approach
# To address the limitations of anecdotes, we will use the tools of statistics, which include:
#     Data collection:
#       We will use data from a large national survey that was designed explicitly with the goal of
#           generating statistically valid inferences about the U.S. population.
#     Descriptive statistics:
#       We will generate statistics that summarize the data concisely, and evaluate different ways to visualize data.
#     Exploratory data analysis:
#       We will look for patterns, differences, and other features that address the questions we are interested in.
#       At the same time we will check for inconsistencies and identify limitations.
#     Estimation:
#       We will use data from a sample to estimate characteristics of the general population.
#     Hypothesis testing:
#       Where we see apparent effects, like a difference between two groups,
#           we will evaluate whether the effect might have happened by chance.

# By performing these steps with care to avoid pitfalls,
#   we can reach conclusions that are more justifiable and more likely to be correct.


# endregion 1.1  A statistical approach

# region 1.2  The National Survey of Family Growth
# codebook:
#   documents the design of the study, the survey questions, and the encoding of the responses.


# endregion 1.2  The National Survey of Family Growth

# region 1.4  DataFrames
# DataFrame,
#   is the fundamental data structure provided by pandas,
#       which is a Python data and statistics package.
#   A DataFrame contains
#       a row for each record,
#           in this case one row per pregnancy, and
#       a column for each variable.
#   In addition to the data, a DataFrame also contains
#       the variable names and their types, and
#       it provides methods for accessing and modifying the data.

# import files from different folder.
# needs __init__.py empty files in all project folders.
from code import nsfg
df = nsfg.ReadFemPreg()

# get a truncated view of the rows and columns, and the shape of the DataFrame
print(df)

# attribute columns returns a sequence of column names as Unicode strings:
# result is an Index, which is another pandas data structure
print(df.columns)
print(df.columns[1])

# To access a column from a DataFrame, you can use the column name as a key:
pregordr = df["pregordr"]

# You can also access the columns of a DataFrame using dot notation:
pregordr = df.pregordr

# This notation only works if the column name is a valid Python identifier,
#   so it has to begin with a letter, can’t contain spaces, etc.


print(type(pregordr))

# The result is a Series, yet another pandas data structure.
# A Series is like a Python list with some additional features.
# When you print a Series, you get the indices and the corresponding values:
print(pregordr)

# In this example the indices are integers from 0 to 13592, but in general they can be any sortable type.
# The elements are also integers, but they can be any type.

# The last line includes the variable name, Series length, and data type;
#   int64 is one of the types provided by NumPy.
#   If you run this example on a 32-bit machine you might see int32.

# You can access the elements of a Series using integer indices and slices:
print(pregordr[0])
print(pregordr[2:5])

# The result of the index operator is an int64;
# the result of the slice is another Series.


# endregion 1.4  DataFrames

# region 1.5  Variables
# Recodes are often based on logic that checks the consistency and accuracy of the data.
# In general it is a good idea to use recodes when they are available,
#   unless there is a compelling reason to process the raw data yourself.


# endregion 1.5  Variables

# region 1.6  Transformation
# replace method
#   replaces these values with np.nan,
#       a special floating-point value that represents “not a number.”
#   The inplace flag
#       tells replace to modify the existing Series rather than create a new one.

# As part of the IEEE floating-point standard, all mathematical operations return nan if either argument is nan:
import numpy as np
np.nan / 100.0

# when you add a new column to a DataFrame, you must use dictionary syntax, like this
# CORRECT
df['totalwgt_lb'] = df.birthwgt_lb + df.birthwgt_oz / 16.0

# Not dot notation, like this:
# WRONG!
df.totalwgt_lb = df.birthwgt_lb + df.birthwgt_oz / 16.0

# The version with dot notation adds an attribute to the DataFrame object, but that attribute is not treated as a new column.


# endregion 1.6  Transformation

# region 1.7  Validation
# One way to validate data is to compute basic statistics and compare them with published results.
# example,
#   the NSFG codebook includes tables that summarize each variable. Here is the table for outcome, which encodes the outcome of each pregnancy:

# The Series class provides a method, value_counts,
#   that counts the number of times each value appears.
# If we select the outcome Series from the DataFrame, we can use value_counts to compare with the published data:
df.outcome.value_counts().sort_index()

# The result of value_counts is a Series;
# sort_index()
#   sorts the Series by index, so the values appear in order.

# Similarly, here is the published table for birthwgt_lb
df.birthwgt_lb.value_counts(sort=False)
# The counts for 6, 7, and 8 pounds check out, and if you add up the counts for 0-5 and 9-95, they check out, too.
# But if you look more closely, you will notice one value that has to be an error, a 51 pound baby!

# To deal with this error, I added a line to CleanFemPreg:
df.loc[df.birthwgt_lb > 20, 'birthwgt_lb'] = np.nan

# This statement replaces invalid values with np.nan.
# The attribute loc
#   provides several ways to select rows and columns from a DataFrame.
#   In this example,
#       the first expression in brackets is the row indexer;
#       the second expression selects the column.

# The expression df.birthwgt_lb > 20 yields a Series of type bool, where True indicates that the condition is true.
# When a boolean Series is used as an index,
#   it selects only the elements that satisfy the condition.


# endregion 1.7  Validatio

# region 1.8  Interpretation
# To work with data effectively, you have to think on two levels at the same time:
#   the level of statistics and
#   the level of context.

# As an example,
#   let’s look at the sequence of outcomes for a few respondents.
#   Because of the way the data files are organized, we have to do some processing to collect the pregnancy data for each respondent.
#   Here’s a function that does that:
def MakePregMap(df):
    d = defaultdict(list)
    for index, caseid in df.caseid.iteritems():
        d[caseid].append(index)
    return d

# df is the DataFrame with pregnancy data.
# The iteritems method
#   enumerates the index (row number) and caseid for each pregnancy.

# d is a dictionary that maps from each case ID to a list of indices.
# If you are not familiar with defaultdict, it is in the Python collections module.
# Using d, we can look up a respondent and get the indices of that respondent’s pregnancies.
caseid = 10229
preg_map = nsfg.MakePregMap(df)
indices = preg_map[caseid]
print(df.outcome[indices].values)


# indices is the list of indices for pregnancies corresponding to respondent 10229.

# Using this list as an index into df.outcome selects the indicated rows and yields a Series.
# Instead of printing the whole Series, I selected the values attribute,
#   which is a NumPy array.


# endregion 1.8  Interpretation

# region 1.9  Exercises
# region Exercise 1
#   file - chap01ex.ipynb


# endregion Exercise 1

# region Exercise 2
#   chap01ex.py


# endregion Exercise 2

# region Exercise 3
# Governments are good sources because data from public research is often freely available.
# Good places to start include
#   http://www.data.gov/, and
#   http://www.science.gov/, and
#   in the United Kingdom, http://data.gov.uk/.
#   General Social Survey at http://www3.norc.org/gss+website/, and
#   the European Social Survey at http://www.europeansocialsurvey.org/.


# endregion Exercise 3


# endregion 1.9  Exercises

# region 1.10  Glossary
#     anecdotal evidence:
#       Evidence, often personal, that is collected casually rather than by a well-designed study.
#     population:
#       A group we are interested in studying.
#       “Population” often refers to a group of people, but the term is used for other subjects, too.
#     cross-sectional study:
#       A study that collects data about a population at a particular point in time.
#     cycle:
#       In a repeated cross-sectional study, each repetition of the study is called a cycle.
#     longitudinal study:
#       A study that follows a population over time, collecting data from the same group repeatedly.
#     record:
#       In a dataset, a collection of information about a single person or other subject.
#     respondent:
#       A person who responds to a survey.
#     sample:
#       The subset of a population used to collect data.
#     representative:
#       A sample is representative if every member of the population has the same chance of being in the sample.
#     oversampling:
#       The technique of increasing the representation of a sub-population in order to avoid errors due to small sample sizes.
#     raw data:
#       Values collected and recorded with little or no checking, calculation or interpretation.
#     recode:
#       A value that is generated by calculation and other logic applied to raw data.
#     data cleaning:
#       Processes that include validating data, identifying errors, translating between data types and representations, etc.

# endregion 1.10  Glossary

# endregion Chapter 1  Exploratory data analysis

# region Chapter 2  Distributions
# region 2.1  Histograms


# endregion 2.1  Histograms


# endregion Chapter 2  Distributions
