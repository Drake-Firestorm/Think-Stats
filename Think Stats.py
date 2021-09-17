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
# file - chap02soln.ipynb

# endregion Exercise 2

# region Exercise 3
# chap02ex.py

# endregion Exercise 3


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


# endregion Chapter 3  Probability mass functions
