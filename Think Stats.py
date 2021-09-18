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
# In the repository you downloaded, you should find a file named chap01ex.ipynb, which is an IPython notebook.
# You can launch IPython notebook from the command line like this:
$ ipython notebook &

# If IPython is installed, it should launch a server that runs in the background and open a browser to view the notebook.
# If you are not familiar with IPython, I suggest you start at http://ipython.org/ipython-doc/stable/notebook/notebook.html.

# To launch the IPython notebook server, run:
$ ipython notebook &

# It should open a new browser window, but if not,
#   the startup message provides a URL you can load in a browser, usually http://localhost:8888.
# The new window should list the notebooks in the repository.

# Open chap01ex.ipynb. Some cells are already filled in, and you should execute them.
# Other cells give you instructions for exercises you should try.

# A solution to this exercise is in chap01soln.ipynb

# ============================================================================================================

#   File - chap01ex.ipynb


# endregion Exercise 1

# region Exercise 2
# In the repository you downloaded, you should find a file named chap01ex.py;
# using this file as a starting place, write a function that reads the respondent file, 2002FemResp.dat.gz.

# The variable pregnum is a recode that indicates how many times each respondent has been pregnant.
# Print the value counts for this variable and compare them to the published results in the NSFG codebook.

# You can also cross-validate the respondent and pregnancy files by
#   comparing pregnum for each respondent with the number of records in the pregnancy file.

# You can use nsfg.MakePregMap to make a dictionary
#   that maps from each caseid to a list of indices into the pregnancy DataFrame.

# A solution to this exercise is in chap01soln.py

# ============================================================================================================

#   File - chap01ex.py


# endregion Exercise 2

# region Exercise 3
# The best way to learn about statistics is to work on a project you are interested in.
# Is there a question like, “Do first babies arrive late,” that you want to investigate?

# Think about questions
#   you find personally interesting, or
#   items of conventional wisdom, or
#   controversial topics, or
#   questions that have political consequences,
#  and see if you can formulate a question that lends itself to statistical inquiry.

# Look for data to help you address the question.

# Governments are good sources because data from public research is often freely available.
# Good places to start include
#   http://www.data.gov/, and
#   http://www.science.gov/, and
#   in the United Kingdom, http://data.gov.uk/.
#   General Social Survey at http://www3.norc.org/gss+website/, and
#   the European Social Survey at http://www.europeansocialsurvey.org/.

# If it seems like someone has already answered your question, look closely to see whether the answer is justified.
# There might be flaws in the data or the analysis that make the conclusion unreliable.
# In that case you could perform a different analysis of the same data, or look for a better source of data.

# If you find a published paper that addresses your question, you should be able to get the raw data.
# Many authors make their data available on the web,
#   but for sensitive data you might have to write to the authors,
#   provide information about how you plan to use the data,
#   or agree to certain terms of use.
# Be persistent!


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
# In Python, an efficient way to compute frequencies is with a dictionary.
# Given a sequence of values, t:
hist = {}
for x in t:
    hist[x] = hist.get(x, 0) + 1

# The result is a dictionary that maps from values to frequencies.
# Alternatively, you could use the Counter class defined in the collections module:
from collections import Counter
counter = Counter(t)

# The result is a Counter object, which is a subclass of dictionary.

# Another option is to use the pandas method value_counts, which we saw in the previous chapter.

# But for this book I created a class, Hist, that represents histograms and provides the methods that operate on them.


# endregion 2.1  Histograms

# region 2.2  Representing histograms
# The Hist constructor can take a sequence, dictionary, pandas Series, or another Hist.
# You can instantiate a Hist object like this:
import thinkstats2
hist = thinkstats2.Hist([1, 2, 2, 3, 5])

# Hist objects provide Freq, which takes a value and returns its frequency:
hist.Freq(2)

# The bracket operator does the same thing:
hist[2]

# If you look up a value that has never appeared, the frequency is 0.
hist.Freq(4)

# Values returns an unsorted list of the values in the Hist:
hist.Values()

# To loop through the values in order, you can use the built-in function sorted:
for val in sorted(hist.Values()):
    print(val, hist.Freq(val))

# Or you can use Items to iterate through value-frequency pairs:
for val, freq in hist.Items():
    print(val, freq)


# endregion 2.2  Representing histograms

# region 2.3  Plotting histograms
# I wrote a module called thinkplot.py that provides functions for plotting Hists and other objects defined in thinkstats2.py.
# It is based on pyplot, which is part of the matplotlib package.

# To plot hist with thinkplot, try this:
import thinkplot
thinkplot.Hist(hist)
thinkplot.Show(xlabel="value", ylabel="frequency")


# endregion 2.3  Plotting histograms

# region 2.4  NSFG variables
# When you start working with a new dataset,
#   I suggest you explore the variables you are planning to use one at a time, and
#   a good way to start is by looking at histograms.

# I’ll start by reading the data and selecting records for live births:
preg = nsfg.ReadFemPreg()
live = preg[preg.outcome == 1]

# The expression in brackets is a boolean Series that selects rows from the DataFrame and returns a new DataFrame.
# Next I generate and plot the histogram of birthwgt_lb for live births.
hist = thinkstats2.Hist(live.birthwgt_lb, label="birthwgt_lb")
thinkplot.Hist(hist)
thinkplot.Show(xlabel="pounds", ylabel="frequency")

# When the argument passed to Hist is a pandas Series, any nan values are dropped.
# label is a string that appears in the legend when the Hist is plotted.

# The most common value, called the mode, is 7 pounds.
# The distribution is approximately bell-shaped,
#   which is the shape of the normal distribution,
#   also called a Gaussian distribution.
# But unlike a true normal distribution, this distribution is asymmetric;
#   it has a tail that extends farther to the left than to the right.

hist = thinkstats2.Hist(live.birthwgt_oz, label="birthwgt_oz")
thinkplot.Hist(hist)
thinkplot.Show(xlabel="ounces", ylabel="frequency")

# histogram of birthwgt_oz, which is the ounces part of birth weight.
# In theory we expect this distribution to be uniform;
#   that is, all values should have the same frequency.
#   In fact, 0 is more common than the other values, and 1 and 15 are less common,
#       probably because respondents round off birth weights that are close to an integer value.

hist = thinkstats2.Hist(np.floor(live.agepreg), label="agepreg")
thinkplot.Hist(hist)
thinkplot.Show(xlabel="years", ylabel="frequency")

# the histogram of agepreg, the mother’s age at the end of pregnancy.
# The mode is 21 years.
# The distribution is very roughly bell-shaped, but in this case the tail extends farther to the right than left;
#   most mothers are in their 20s, fewer in their 30s.

hist = thinkstats2.Hist(np.ceil(live.prglngth), label="prglngth")
thinkplot.Hist(hist)
thinkplot.Show(xlabel="weeks", ylabel="frequency")

# the histogram of prglngth, the length of the pregnancy in weeks.
# By far the most common value is 39 weeks.
# The left tail is longer than the right;
#   early babies are common, but
#   pregnancies seldom go past 43 weeks, and doctors often intervene if they do.


# endregion 2.4  NSFG variables

# region 2.5  Outliers
# Hist provides methods Largest and Smallest, which take an integer n and return the n largest or smallest values from the histogram:
for weeks, freq in hist.Smallest(10):
    print(weeks, freq)

# The best way to handle outliers depends on
#   “domain knowledge”;
#       that is, information about where the data come from and what they mean. And it depends on
#       what analysis you are planning to perform.


# endregion 2.5  Outliers

# region 2.6  First babies
# compare the distribution of pregnancy lengths for first babies and others.
# I divided the DataFrame of live births using birthord, and computed their histograms:
firsts = live[live.birthord == 1]
others = live[live.birthord != 1]

first_hist = thinkstats2.Hist(firsts.prglngth, label="first")
others_hist = thinkstats2.Hist(others.prglngth, label="other")

# Then I plotted their histograms on the same axis:
width = 0.45
thinkplot.PrePlot(2)
thinkplot.Hist(first_hist, align="right", width=width)
thinkplot.Hist(others_hist, align="left", width=width)
thinkplot.Show(xlabel="weeks", ylabel="frequency", xlim=[27, 46])

# thinkplot.PrePlot takes the number of histograms we are planning to plot;
#   it uses this information to choose an appropriate collection of colors.

# thinkplot.Hist normally uses align='center' so that each bar is centered over its value.
#   For this figure, I use align='right' and align='left' to place corresponding bars on either side of the value.

# With width=0.45, the total width of the two bars is 0.9, leaving some space between each pair.

# Finally, I adjust the axis to show only data between 27 and 46 weeks.

# Histograms are useful because they make the most frequent values immediately apparent.
#   But they are not the best choice for comparing two distributions.
# In this example,
#   there are fewer “first babies” than “others,” so some of the apparent differences in the histograms are due to sample sizes.
#   In the next chapter we address this problem using probability mass functions.


# endregion 2.6  First babies

# region 2.7  Summarizing distributions
# A histogram is a complete description of the distribution of a sample;
#   that is, given a histogram, we could reconstruct the values in the sample (although not their order).

# If the details of the distribution are important, it might be necessary to present a histogram.
# But often we want to summarize the distribution with a few descriptive statistics.

# Some of the characteristics we might want to report are:
#     central tendency:
#       Do the values tend to cluster around a particular point?
#     modes:
#       Is there more than one cluster?
#     spread:
#       How much variability is there in the values?
#     tails:
#       How quickly do the probabilities drop off as we move away from the modes?
#     outliers:
#       Are there extreme values far from the modes?

# Statistics designed to answer these questions are called summary statistics.
# By far the most common summary statistic is the mean,
#   which is meant to describe the central tendency of the distribution.

# If you have a sample of n values, xi, the mean, x, is the sum of the values divided by the number of values; in other words
# x_bar = 	1 / n * ∑i xi

#  The words “mean” and “average” are sometimes used interchangeably, but I make this distinction:
#     The “mean” of a sample is the summary statistic computed with the previous formula.
#     An “average” is one of several summary statistics you might choose to describe a central tendency.

# Sometimes the mean is a good description of a set of values.
# For example,
#   apples are all pretty much the same size (at least the ones sold in supermarkets).
#       So if I buy 6 apples and the total weight is 3 pounds,
#           it would be a reasonable summary to say they are about a half pound each.
#   But pumpkins are more diverse.
#       Suppose I grow several varieties in my garden, and one day I harvest
#           three decorative pumpkins that are 1 pound each,
#           two pie pumpkins that are 3 pounds each, and
#           one Atlantic Giant® pumpkin that weighs 591 pounds.
#       The mean of this sample is 100 pounds,
#           but if I told you “The average pumpkin in my garden is 100 pounds,” that would be misleading.
#       In this example, there is no meaningful average because there is no typical pumpkin.


# endregion 2.7  Summarizing distributions

# region 2.8  Variance
# If there is no single number that summarizes pumpkin weights, we can do a little better with two numbers: mean and variance.

# Variance is a summary statistic intended to describe the variability or spread of a distribution. The variance of a set of values is
# S**2 = 1 / n * ∑i (xi − x_bar)**2

#  The term xi − x_bar is called the “deviation from the mean,” so variance is the mean squared deviation.
#  The square root of variance, S, is the standard deviation.

# If you have prior experience, you might have seen a formula for variance with n−1 in the denominator, rather than n.
# This statistic is used to estimate the variance in a population using a sample.
# We will come back to this in Chapter 8.

# Pandas data structures provides methods to compute mean, variance and standard deviation:
mean = live.prglngth.mean()
var = live.prglngth.var()
std = live.prglngth.std()

# For all live births,
#   the mean pregnancy length is 38.6 weeks,
#   the standard deviation is 2.7 weeks,
#       which means we should expect deviations of 2-3 weeks to be common.

# Variance of pregnancy length is 7.3,
#   which is hard to interpret,
#       especially since the units are weeks2, or “square weeks.”
# Variance is useful in some calculations,
#   but it is not a good summary statistic.


# endregion 2.8  Variance

# region 2.9  Effect size
# An effect size is a summary statistic intended to describe (wait for it) the size of an effect.
# example,
#   to describe the difference between two groups, one obvious choice is the difference in the means.

# Another way to convey the size of the effect is to compare the difference between groups to the variability within groups.
# Cohen’s d is a statistic intended to do that; it is defined
# d = 	x_1_bar − x_2_bar / s
#   where
#       x_1_bar and x_2_bar are the means of the groups and
#       s is the “pooled standard deviation”.

# Here’s the Python code that computes Cohen’s d:
def CohenEffectSize(group1, group2):
    diff = group1.mean() - group2.mean()

    var1 = group1.var()
    var2 = group2.var()
    n1, n2 = len(group1), len(group2)

    pooled_var = (n1 * var1 + n2 * var2) / (n1 + n2)
    d = diff / math.sqrt(pooled_var)
    return d

#  See - https://en.wikipedia.org/wiki/Effect_size


# endregion 2.9  Effect size

# region 2.10  Reporting results
# We have seen several ways to describe the difference in pregnancy length (if there is one) between first babies and others.
# How should we report these results?

# The answer depends on who is asking the question.
#   A scientist might be interested in any (real) effect, no matter how small.
#   A doctor might only care about effects that are clinically significant;
#       that is, differences that affect treatment decisions.
#   A pregnant woman might be interested in results that are relevant to her,
#       like the probability of delivering early or late.

# How you report results also depends on your goals.
#   If you are trying to demonstrate the importance of an effect,
#       you might choose summary statistics that emphasize differences.
#   If you are trying to reassure a patient,
#       you might choose statistics that put the differences in context.

# Of course your decisions should also be guided by professional ethics.
# It’s ok to be persuasive; you should design statistical reports and visualizations that tell a story clearly.
# But you should also do your best to make your reports honest, and to acknowledge uncertainty and limitations.


# endregion 2.10  Reporting results

# region 2.11  Exercises
# region Exercise 1
# Based on the results in this chapter,
#   suppose you were asked to summarize what you learned about whether first babies arrive late.

# Which summary statistics would you use if you wanted to get a story on the evening news?
# Which ones would you use if you wanted to reassure an anxious patient?

# Finally, imagine that you are Cecil Adams, author of The Straight Dope (http://straightdope.com),
# and your job is to answer the question, “Do first babies arrive late?”
# Write a paragraph that uses the results in this chapter to answer the question clearly, precisely, and honestly.

# ============================================================================================================

# Evening news - summary statistics (mean, std)
# reassure an anxious patient - Effect size (Cohen's d)

# For all live births, the mean pregnancy length is 38.6 weeks,
# the standard deviation is 2.7 weeks, which means we should expect deviations of 2-3 weeks to be common.
# Mean pregnancy length for first babies is 38.601; for other babies it is 38.523.
# The difference is 0.078 weeks, which works out to 13 hours.
# As a fraction of the typical pregnancy length, this difference is about 0.2%.
# Finaly, the difference in means is 0.029 standard deviations, which is small.
# To put that in perspective, the difference in height between men and women is about 1.7 standard deviations
# However, the analysis did reveal outliers with certain births taking less than 30 weeks and others more than 42 weeks.
# Based on the above analysis, there is little difference between the arrival of first babies and the subsequent babies.
# But on a case by case basis there might be difference from the norm.


# endregion Exercise 1

# region Exercise 2
# In the repository you downloaded, you should find a file named chap02ex.ipynb; open it.
# Some cells are already filled in, and you should execute them.
# Other cells give you instructions for exercises.
# Follow the instructions and fill in the answers.

# A solution to this exercise is in chap02soln.ipynb

# ============================================================================================================

#   File - chap02soln.ipynb

# endregion Exercise 2

# In the repository you downloaded, you should find a file named chap02ex.py;
# you can use this file as a starting place for the following exercises. My solution is in chap02soln.py.

# region Exercise 3
# The mode of a distribution is the most frequent value; see http://wikipedia.org/wiki/Mode_(statistics).
# Write a function called Mode that takes a Hist and returns the most frequent value.

# As a more challenging exercise,
#   write a function called AllModes that returns a list of value-frequency pairs in descending order of frequency.

# ============================================================================================================

#   File - chap02ex.py

# endregion Exercise 3

# region Exercise 4
# Using the variable totalwgt_lb, investigate whether first babies are lighter or heavier than others.
# Compute Cohen’s d to quantify the difference between the groups.
# How does it compare to the difference in pregnancy length?

# ============================================================================================================

#   File - chap02ex.py

# endregion Exercise 4

# endregion 2.11  Exercises

# region 2.12  Glossary
#     distribution:
#       The values that appear in a sample and the frequency of each.
#     histogram:
#       A mapping from values to frequencies, or a graph that shows this mapping.
#     frequency:
#       The number of times a value appears in a sample.
#     mode:
#       The most frequent value in a sample, or one of the most frequent values.
#     normal distribution:
#       An idealization of a bell-shaped distribution; also known as a Gaussian distribution.
#     uniform distribution:
#       A distribution in which all values have the same frequency.
#     tail:
#       The part of a distribution at the high and low extremes.
#     central tendency:
#       A characteristic of a sample or population; intuitively, it is an average or typical value.
#     outlier:
#       A value far from the central tendency.
#     spread:
#       A measure of how spread out the values in a distribution are.
#     summary statistic:
#       A statistic that quantifies some aspect of a distribution, like central tendency or spread.
#     variance:
#       A summary statistic often used to quantify spread.
#     standard deviation:
#       The square root of variance, also used as a measure of spread.
#     effect size:
#       A summary statistic intended to quantify the size of an effect like a difference between groups.
#     clinically significant:
#       A result, like a difference between groups, that is relevant in practice.

# endregion 2.12  Glossary


# endregion Chapter 2  Distributions

# region Chapter 3  Probability mass functions
# The code for this chapter is in probability.py.
# For information about downloading and working with this code, see Section 0.2.

# region 3.1  Pmfs
# Given a Hist, we can make a dictionary that maps from each value to its probability:
n = hist.Total()
d = {}
for x, freq in hist.Items():
    d[x] = freq / n

# Or we can use the Pmf class provided by thinkstats2.
# Like Hist, the Pmf constructor can take a list, pandas Series, dictionary, Hist, or another Pmf object.
# Here’s an example with a simple list:
import thinkstats2
pmf = thinkstats2.Pmf([1,2,2,3,5])

# The Pmf is normalized so total probability is 1.

# Pmf and Hist objects are similar in many ways; in fact, they inherit many of their methods from a common parent class.
# example,
#   the methods Values and Items work the same way for both.
#   The biggest difference is that
#       a Hist maps from values to integer counters;
#       a Pmf maps from values to floating-point probabilities.

# To look up the probability associated with a value, use Prob:
pmf.Prob(2)

# The bracket operator is equivalent:
pmf[2]

# You can modify an existing Pmf by incrementing the probability associated with a value:
pmf.Incr(2, 0.2)
pmf.Prob(2)

# Or you can multiply a probability by a factor:
pmf.Mult(2, 0.2)
pmf.Prob(2)

# If you modify a Pmf, the result may not be normalized;
#   that is, the probabilities may no longer add up to 1.
# To check, you can call Total, which returns the sum of the probabilities:
pmf.Total()

# To renormalize, call Normalize:
pmf.Normalize()
pmf.Total()

# Pmf objects provide a Copy method so you can make and modify a copy without affecting the original.

# My notation in this section might seem inconsistent, but there is a system:
#   I use Pmf for the name of the class,
#   pmf for an instance of the class, and
#   PMF for the mathematical concept of a probability mass function.


# endregion 3.1  Pmfs

# region 3.2  Plotting PMFs
# thinkplot provides two ways to plot Pmfs:
#     To plot a Pmf as a bar graph, you can use thinkplot.Hist.
#       Bar graphs are most useful if the number of values in the Pmf is small.
#     To plot a Pmf as a step function, you can use thinkplot.Pmf.
#       This option is most useful if there are a large number of values and the Pmf is smooth.
#       This function also works with Hist objects.

# In addition,
#   pyplot provides a function called hist that takes a sequence of values, computes a histogram, and plots it.
# Since I use Hist objects, I usually don’t use pyplot.hist.
live, firsts, others = first.MakeFrames()

first_pmf = thinkstats2.Pmf(firsts.prglngth)
other_pmf = thinkstats2.Pmf(others.prglngth)

width = 0.45


thinkplot.PrePlot(2, cols=2)
thinkplot.Hist(first_pmf, align="right", width=width)
thinkplot.Hist(other_pmf, align="left", width=width)
thinkplot.Config(xlabel="weeks", ylabel="probability", axis=[27, 46, 0, 0.6])

thinkplot.PrePlot(2)
thinkplot.SubPlot(2)
thinkplot.Pmfs([first_pmf, other_pmf])
thinkplot.Config(xlabel="weeks", axis=[27, 46, 0, 0.6])

thinkplot.Show()


# PrePlot takes optional parameters rows and cols to make a grid of figures, in this case one row of two figures.
# The first figure (on the left) displays the Pmfs using thinkplot.Hist, as we have seen before.

# The second call to PrePlot resets the color generator.
# Then SubPlot switches to the second figure (on the right) and displays the Pmfs using thinkplot.Pmfs.
# I used the axis option to ensure that the two figures are on the same axes,
#   which is generally a good idea if you intend to compare two figures.


# endregion 3.2  Plotting PMFs

# region 3.3  Other visualizations
# Histograms and PMFs are useful while you are exploring data and trying to identify patterns and relationships.
# Once you have an idea what is going on,
#   a good next step is to design a visualization that makes the patterns you have identified as clear as possible.

# In the NSFG data, the biggest differences in the distributions are near the mode.
# So it makes sense to zoom in on that part of the graph, and to transform the data to emphasize differences:
live, firsts, others = first.MakeFrames()

first_pmf = thinkstats2.Pmf(firsts.prglngth)
other_pmf = thinkstats2.Pmf(others.prglngth)

weeks = range(36, 45)
diffs = []
for week in weeks:
    p1 = first_pmf.Prob(week)
    p2 = other_pmf.Prob(week)
    diff = 100 * (p1 - p2)
    diffs.append(diff)

thinkplot.Bar(weeks, diffs)

thinkplot.Show()

# In this code,
#   weeks is the range of weeks;
#   diffs is the difference between the two PMFs in percentage points.
# Figure 3.2 shows the result as a bar chart.

# For now we should hold this conclusion only tentatively.
# We used the same dataset to identify an apparent difference and
#   then chose a visualization that makes the difference apparent.
# We can’t be sure this effect is real; it might be due to random variation.
# We’ll address this concern later.


# endregion 3.3  Other visualizations

# region 3.4  The class size paradox
def BiasPmf(pmf, label):
    new_pmf = pmf.Copy(label=label)

    for x, p in pmf.Items():
        new_pmf.Mult(x, x)

    new_pmf.Normalize()
    return new_pmf


def UnbiasPmf(pmf, label):
    new_pmf = pmf.Copy(label=label)

    for x, p in pmf.Items():
        new_pmf.Mult(x, 1.0/x)

    new_pmf.Normalize()
    return new_pmf


d = {7: 8, 12: 8, 17: 14, 22: 4,
     27: 6, 32: 12, 37: 8, 42: 3, 47: 2}

pmf = thinkstats2.Pmf(d, label='actual')
print('mean', pmf.Mean())

biased_pmf = BiasPmf(pmf, label="observed")
print('mean', biased_pmf.Mean())

unbiased_pmf = UnbiasPmf(biased_pmf, label="original")
print("mean", unbiased_pmf.Mean())

thinkplot.PrePlot(2)
thinkplot.Pmfs([pmf, biased_pmf])
thinkplot.Show(xlabel="class size", ylabel="PMF")


# endregion 3.4  The class size paradox

# region 3.5  DataFrame indexing
# In Section 1.4 we read a pandas DataFrame and used it to select and modify data columns.
# Now let’s look at row selection.
# To start, I create a NumPy array of random numbers and use it to initialize a DataFrame:
import numpy as np
import pandas
array = np.random.randn(4, 2)
df = pandas.DataFrame(array)

# By default, the rows and columns are numbered starting at zero, but you can provide column names:
columns = ["A", "B"]
df = pandas.DataFrame(array, columns=columns)

# You can also provide row names.
#   The set of row names is called the index;
#   the row names themselves are called labels.
index = ["a", "b", "c", "d"]
df = pandas.DataFrame(array, columns=columns, index=index)

# As we saw in the previous chapter, simple indexing selects a column, returning a Series:
df["A"]

# To select a row by label, you can use the loc attribute, which returns a Series:
df.loc["a"]

# If you know the integer position of a row, rather than its label,
#   you can use the iloc attribute, which also returns a Series.
df.iloc[0]

# loc can also take a list of labels; in that case, the result is a DataFrame.
indices = ["a", "c"]
df.loc[indices]

# Finally, you can use a slice to select a range of rows by label:
df.loc["a":"c"]

# Or by integer position:
df.iloc[0:2]

# The result in either case is a DataFrame,
#   but notice that the first result includes the end of the slice;
#   the second doesn’t.

# My advice:
#   if your rows have labels that are not simple integers,
#   use the labels consistently and avoid using integer positions.


# endregion 3.5  DataFrame indexing

# region 3.6  Exercises
# Solutions to these exercises are in chap03soln.ipynb and chap03soln.py

# region Exercise 1
# Something like the class size paradox appears if you survey children and ask how many children are in their family.
# Families with many children are more likely to appear in your sample,
#   and families with no children have no chance to be in the sample.

# Use the NSFG respondent variable NUMKDHH to construct the actual distribution for the number of children under 18 in the household.

# Now compute the biased distribution we would see if we surveyed the children and asked them how many children under 18 (including themselves) are in their household.

# Plot the actual and biased distributions, and compute their means.
# As a starting place, you can use chap03ex.ipynb.

# ============================================================================================================

#   File - chap03ex.ipynb


# endregion Exercise 1

# region Exercise 2
#
# In Section 2.7 we computed the mean of a sample by adding up the elements and dividing by n.
# If you are given a PMF, you can still compute the mean, but the process is slightly different:
# x_bar = ∑i pi * xi
#   where
#       the xi are the unique values in the PMF and
#       pi=PMF(xi).

# Similarly, you can compute variance like this:
# S**2 = ∑i pi * (xi − x_bar)**2

# Write functions called PmfMean and PmfVar that take a Pmf object and compute the mean and variance.
# To test these methods, check that they are consistent with the methods Mean and Var provided by Pmf.

# ============================================================================================================

#   File - Pmf_Functions.py


# endregion Exercise 2

# region Exercise 3
# I started with the question, “Are first babies more likely to be late?”
# To address it, I computed the difference in means between groups of babies,
#   but I ignored the possibility that there might be a difference between first babies and others for the same woman.

# To address this version of the question,
#   select respondents who have at least two babies and compute pairwise differences.
# Does this formulation of the question yield a different result?

# Hint: use nsfg.MakePregMap.

# ============================================================================================================

#   File - chap03ex.ipynb


# endregion Exercise 3

# region Exercise 4
# In most foot races, everyone starts at the same time.
# If you are a fast runner, you usually pass a lot of people at the beginning of the race,
#   but after a few miles everyone around you is going at the same speed.

# When I ran a long-distance (209 miles) relay race for the first time, I noticed an odd phenomenon:
#   when I overtook another runner, I was usually much faster, and
#   when another runner overtook me, he was usually much faster.

# At first I thought that the distribution of speeds might be bimodal;
#   that is, there were many slow runners and many fast runners, but few at my speed.

# Then I realized that I was the victim of a bias similar to the effect of class size.
# The race was unusual in two ways:
#   it used a staggered start, so teams started at different times;
#   also, many teams included runners at different levels of ability.

# As a result, runners were spread out along the course with little relationship between speed and location.
# When I joined the race, the runners near me were (pretty much) a random sample of the runners in the race.

# So where does the bias come from?
# During my time on the course, the chance of overtaking a runner, or being overtaken,
#   is proportional to the difference in our speeds.
# I am more likely to catch a slow runner, and more likely to be caught by a fast runner.
# But runners at the same speed are unlikely to see each other.

# Write a function called ObservedPmf that takes
#   a Pmf representing the actual distribution of runners’ speeds,
#   and the speed of a running observer,
#   and returns a new Pmf representing the distribution of runners’ speeds as seen by the observer.

# To test your function, you can use relay.py,
#   which reads the results from the James Joyce Ramble 10K in Dedham MA and converts the pace of each runner to mph.

# Compute the distribution of speeds you would observe if you ran a relay race at 7.5 mph with this group of runners.
# A solution to this exercise is in relay_soln.py.

# ============================================================================================================

#   File - chap03ex.ipynb


# endregion Exercise 4


# endregion 3.6  Exercises

# region 3.7  Glossary
# Probability mass function (PMF):
#   a representation of a distribution as a function that maps from values to probabilities.
# probability:
#   A frequency expressed as a fraction of the sample size.
# normalization:
#   The process of dividing a frequency by a sample size to get a probability.
# index:
#   In a pandas DataFrame, the index is a special column that contains the row labels.


# endregion 3.7  Glossary

# endregion Chapter 3  Probability mass functions

# region Chapter 4  Cumulative distribution functions


# endregion Chapter 4  Cumulative distribution functions
