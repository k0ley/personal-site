+++ 
date = 2021-04-27T13:46:42-05:00
title = "Predicting the 2020 Presidential Election using Support Vector Regression"
slug = "pres-ml"
categories = []
thumbnail = "images/thumb.png"
description = "Presidential Machine Learning Modeling"
+++

### Introduction
I want to preface this by saying that this is a model that I created in college in one of my Machine Learning classes for a final project in 2016, but I figured it was cool enough to display publicly. Licenced by Tulane University Computer Science.
Now it is 2021, and the 2020 Presidential Election has concluded. Let's see how young Sam did on his ML model compared to what actually happened!

The political landscape in the United States is without a doubt the most divided it has been since possibly the Reconstructionist era after the Civil War. One aspect of the modern political apparatus has remained constant though: there are some states, and even individual COUNTIES, that almost always vote in favor of the winning candidate in the general election. Here's an example: Westmoreland County in Virgia -- a small, agrarian town south of DC -- only failed to vote correctly in the General Election one time since 1928. Keep in mind that this remained constant as Nixon famously shifted once blue states red and vice-versa.

Why does this happen? How is it even possible for one state let alone one county to be so "hyper-intelligent" to be so correct when there are pollsters and massive corporations who spend billions to predict these elections but in the end always seem to slightly (or majorly, as seen in 2016) wrong?

### Data
Training an entire model to focus on just one county is silly. The aforementioned county in Virgia is so sparsely populated that there wouldn't even be enough data to justify what could be seen as a pseudo-ML model.
Instead, I took all of the counties from Pennsylvania (famously a swing state) and decided to train the models based on that.
What do I want to figure out?
- What socioeconomic, geographic, and political structures in an individual househould most accurately predicts how this household will vote?

How will we model this?
- A massive dataset (woooo machine learning!) that accounts for all socioeconomic and geographic statuses of all households in each individual PA county (see below)
- Historical presidential election data in the state of Pennsylvania as well as their individual counties
- President Obama's 2008 election results and Secretary Hillary Clinton's 2012 election results

A few attributes that we are training on (see the rest [here](https://github.com/k0ley/Projects/tree/master/Machine%20Learning/President%20Model)):
|column_name|description                                                        |
|-----------|-------------------------------------------------------------------|
|AGE775214  |Persons 65 years and over, percent, 2014                           |
|SEX255214  |Female persons, percent, 2014                                      |
|RHI125214  |White alone, percent, 2014                                         |
|RHI225214  |Black or African American alone, percent, 2014                     |
|RHI625214  |Two or More Races, percent, 2014                                   |
|RHI825214  |White alone, not Hispanic or Latino, percent, 2014                 |
|HSG445213  |Homeownership rate, 2009-2013                                      |
|HSG495213  |Median value of owner-occupied housing units, 2009-2013            |
|HSD410213  |Households, 2009-2013                                              |
|HSD310213  |Persons per household, 2009-2013                                   |
|INC910213  |Per capita money income in past 12 months (2013 dollars), 2009-2013|
|INC110213  |Median household income, 2009-2013                                 |
|PVY020213  |Persons below poverty level, percent, 2009-2013                    |
|POP060210  |Population per square mile, 2010                                   |

### Why SVR?
I could get into the nitty gritty about the algorithms that make SVR work, but generally speaking one would use SVR to solve the same problems you would solve linear regression with. The key difference is that SVR also allows us to model non-linear relationships between variables and yields more flexibility to adjust the model by fine-tuning hyper-parameters in the code.



### Code!
Importing some necessary modules, defining our features, importing csv data (in retrospect, CSV data is not ideal! if I refactored this today I would do it differently.)
```python
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn import utils

Features = ['fips', 'AGE775214', 'SEX255214', 'RHI125214', 'RHI225214', 'RHI325214',
'RHI425214', 'RHI525214', 'RHI625214', 'RHI725214', 'RHI825214', 'POP715213', 'POP645213',
'EDU685213', 'VET605213', 'LFE305213', 'HSG010214', 'HSG445213', 'HSG495213', 'HSD410213',
'HSD310213', 'INC910213', 'INC110213', 'PVY020213', 'SBO001207', 'LND110210', 'POP060210']

#take in 4 datasets: labels explained, county data,  presidential voting data in 2012, and voting data in 2016
featuresExplained = pd.read_csv("labels.csv")
CountyData = pd.read_csv("county data.csv")
VotingData = pd.read_csv("lineardata.csv")
FutureData = pd.read_csv("clintondata.csv")
```

Next, we want to preprocess our data to reduce outliers, extreme complexity, and just overall train our model in a much cleaner fashion.
```python
S = CountyData.iloc[:,1:27].values
t = VotingData.values
future = FutureData.values

sc_S = preprocessing.StandardScaler()
sc_t = preprocessing.StandardScaler()
sc_2020 = preprocessing.StandardScaler()
```

Lastly, let's train our model! You can see I commented out the poly-algorithm as it gave slightly worse results, but still a very viable option for complex datasets like this.
```python
S2 = sc_S.fit_transform(S)
t2 = sc_t.fit_transform(t)
futureElection = sc_2020.fit_transform(future)

#support vector regression
# clf = SVR(kernel='poly', degree=3, verbose=True) #gamma='auto'
clf = SVR(kernel='linear')
clf.fit(S2, t2)
initial = sc_t.inverse_transform(clf.predict(sc_S.transform(S)))
new = sc_2020.inverse_transform(clf.predict(sc_S.transform(S)))

print(clf.score(S2, futureElection))
```

- 78.4% accuracy for kernel = 'poly'
- 79.8% accuracy for kernel = 'rbf'
- 88.65% accuracy for kernel = 'linear'

### Visualization

Let's chat. Below is a little chart I put together in Jupyter to ascertain our first question: what feature has the most impact on deciding a counties voting shift?
![election-vis](/images/projects/election-ml/chart.png)

The highest weighted coefficients:
- Population
- Median household income
- % of people below the federal poverty level

The lowest weighted coefficients:
- American Indian or Alaskan %
- Mean Travel Time to Work

This code to visualize this data is all public on my GitHub [here](https://github.com/k0ley).

### Final Remarks

My Machine Learning model built in 2016 to predict 2020 GE results for PA, visualized:

![2016](/images/projects/election-ml/2016.png)

The actual election results in 2020 for PA, visualized:

![2020](/images/projects/election-ml/2020.png)


Maybe you can tell, but the model was only wrong about one county! :) (looking at you, erie county)

Not all is good though. Let's talk about how this model could potentially be misleading, as well as give some political-economic insight through some widely accepted theorems.
- **Downs Median Voter Theorem**
    - We are essentially clumping all individual counties as one person. While this works in theory, it directly goes against downs median voter theorem, which stipulates that "if voters and policies are distributed along a one-dimensional spectrum, with voters ranking alternatives in order of proximity, then any voting method which satisfies the Condorcet criterion will elect the candidate closest to the median voter. In particular, a majority vote between two options will do so."
- **Can't account for Voter Turnout**
	- Well of course we could in retrospect, but at the time we had no clue who would turn out! In fact, there is no publicly available historical turnout data or else we would have integrated it into the model.
- **Inherit Bias**
	- Joe Biden built his political career in Delaware. Growing up in South Jersey, the entire tri-state area is essentially politically linked. People from Philly certainly could have voted for Joe Biden as a one-issue voter due to the fact that he is a Phillies fan. Who knows?

### Related Works
[Mohammed Zolghadr, Seyed Armin Akhavan Niaki, S. T. A. Niaki - Modeling and Forecasting US presidential election using learning algorithms](https://www.researchgate.net/publication/319617652_Modeling_and_Forecasting_US_Presidential_Election_Using_Learning_Algorithms)

[Tynan Challenor - Predicting Votes from Census Data](http://cs229.stanford.edu/proj2017/final-reports/5232542.pdf)

