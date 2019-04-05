
# Examining Racial Discrimination in the US Job Market

### Background
Racial discrimination continues to be pervasive in cultures throughout the world. Researchers examined the level of racial discrimination in the United States labor market by randomly assigning identical résumés to black-sounding or white-sounding names and observing the impact on requests for interviews from employers.

### Data
In the dataset provided, each row represents a resume. The 'race' column has two values, 'b' and 'w', indicating black-sounding and white-sounding. The column 'call' has two values, 1 and 0, indicating whether the resume received a call from employers or not.

Note that the 'b' and 'w' values in race are assigned randomly to the resumes when presented to the employer.

### Exercises
You will perform a statistical analysis to establish whether race has a significant impact on the rate of callbacks for resumes.

Answer the following questions **in this notebook below and submit to your Github account**. 

   1. What test is appropriate for this problem? Does CLT apply?
   2. What are the null and alternate hypotheses?
   3. Compute margin of error, confidence interval, and p-value. Try using both the bootstrapping and the frequentist statistical approaches.
   4. Write a story describing the statistical significance in the context or the original problem.
   5. Does your analysis mean that race/name is the most important factor in callback success? Why or why not? If not, how would you amend your analysis?

You can include written notes in notebook cells using Markdown: 
   - In the control panel at the top, choose Cell > Cell Type > Markdown
   - Markdown syntax: http://nestacms.com/docs/creating-content/markdown-cheat-sheet

#### Resources
+ Experiment information and data source: http://www.povertyactionlab.org/evaluation/discrimination-job-market-united-states
+ Scipy statistical methods: http://docs.scipy.org/doc/scipy/reference/stats.html 
+ Markdown syntax: http://nestacms.com/docs/creating-content/markdown-cheat-sheet
+ Formulas for the Bernoulli distribution: https://en.wikipedia.org/wiki/Bernoulli_distribution


```python
import pandas as pd
import numpy as np
from scipy import stats
```


```python
data = pd.io.stata.read_stata('data/us_job_market_discrimination.dta')
```


```python
# number of callbacks for white-sounding names and black-sounding names:
print(sum(data[data.race=='w'].call), sum(data[data.race=='b'].call))
print(len(data[data.race=='w']), len(data[data.race=='b']))
```

    235.0 157.0
    2435 2435



```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>ad</th>
      <th>education</th>
      <th>ofjobs</th>
      <th>yearsexp</th>
      <th>honors</th>
      <th>volunteer</th>
      <th>military</th>
      <th>empholes</th>
      <th>occupspecific</th>
      <th>...</th>
      <th>compreq</th>
      <th>orgreq</th>
      <th>manuf</th>
      <th>transcom</th>
      <th>bankreal</th>
      <th>trade</th>
      <th>busservice</th>
      <th>othservice</th>
      <th>missind</th>
      <th>ownership</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>b</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>17</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td></td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>316</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td></td>
    </tr>
    <tr>
      <th>2</th>
      <td>b</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>19</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td></td>
    </tr>
    <tr>
      <th>3</th>
      <td>b</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>313</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td></td>
    </tr>
    <tr>
      <th>4</th>
      <td>b</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>22</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>313</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>Nonprofit</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 65 columns</p>
</div>




```python
data.columns
```




    Index(['id', 'ad', 'education', 'ofjobs', 'yearsexp', 'honors', 'volunteer',
           'military', 'empholes', 'occupspecific', 'occupbroad', 'workinschool',
           'email', 'computerskills', 'specialskills', 'firstname', 'sex', 'race',
           'h', 'l', 'call', 'city', 'kind', 'adid', 'fracblack', 'fracwhite',
           'lmedhhinc', 'fracdropout', 'fraccolp', 'linc', 'col', 'expminreq',
           'schoolreq', 'eoe', 'parent_sales', 'parent_emp', 'branch_sales',
           'branch_emp', 'fed', 'fracblack_empzip', 'fracwhite_empzip',
           'lmedhhinc_empzip', 'fracdropout_empzip', 'fraccolp_empzip',
           'linc_empzip', 'manager', 'supervisor', 'secretary', 'offsupport',
           'salesrep', 'retailsales', 'req', 'expreq', 'comreq', 'educreq',
           'compreq', 'orgreq', 'manuf', 'transcom', 'bankreal', 'trade',
           'busservice', 'othservice', 'missind', 'ownership'],
          dtype='object')



<div class="span5 alert alert-success">
<p>Your answers to Q1 and Q2 here
    
1. The central limit theorem (CLT) does apply here, as the number of total resumes for each distribution exceeds 2000. That is more than enough to assume the distribution is roughly Gaussian (normal) even if it's not actually. As such, a Z-test is an appropriate statistical test to use.

2. The null hypothesis in this case is that there is no statistical difference between the number of callbacks for white and black sounding names. The alternate hypothesis is that there is a difference.
</p>
</div>

# Question 3:


```python
# Separate data into two sets, the white- and black-sounding name resumes.
w = data[data.race=='w']
b = data[data.race=='b']
```


```python
w.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>education</th>
      <th>ofjobs</th>
      <th>yearsexp</th>
      <th>honors</th>
      <th>volunteer</th>
      <th>military</th>
      <th>empholes</th>
      <th>occupspecific</th>
      <th>occupbroad</th>
      <th>workinschool</th>
      <th>...</th>
      <th>educreq</th>
      <th>compreq</th>
      <th>orgreq</th>
      <th>manuf</th>
      <th>transcom</th>
      <th>bankreal</th>
      <th>trade</th>
      <th>busservice</th>
      <th>othservice</th>
      <th>missind</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2435.000000</td>
      <td>2435.000000</td>
      <td>2435.000000</td>
      <td>2435.000000</td>
      <td>2435.000000</td>
      <td>2435.000000</td>
      <td>2435.000000</td>
      <td>2435.000000</td>
      <td>2435.000000</td>
      <td>2435.000000</td>
      <td>...</td>
      <td>2435.000000</td>
      <td>2435.000000</td>
      <td>2435.00000</td>
      <td>2435.000000</td>
      <td>2435.000000</td>
      <td>2435.000000</td>
      <td>2435.000000</td>
      <td>2435.000000</td>
      <td>2435.000000</td>
      <td>2435.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.620945</td>
      <td>3.664476</td>
      <td>7.856263</td>
      <td>0.054209</td>
      <td>0.408624</td>
      <td>0.092402</td>
      <td>0.450103</td>
      <td>214.530595</td>
      <td>3.475154</td>
      <td>0.558111</td>
      <td>...</td>
      <td>0.106776</td>
      <td>0.436961</td>
      <td>0.07269</td>
      <td>0.082957</td>
      <td>0.030390</td>
      <td>0.085010</td>
      <td>0.213963</td>
      <td>0.267762</td>
      <td>0.154825</td>
      <td>0.165092</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.696609</td>
      <td>1.219345</td>
      <td>5.079228</td>
      <td>0.226477</td>
      <td>0.491681</td>
      <td>0.289653</td>
      <td>0.497606</td>
      <td>148.255302</td>
      <td>2.033334</td>
      <td>0.496714</td>
      <td>...</td>
      <td>0.308892</td>
      <td>0.496112</td>
      <td>0.25968</td>
      <td>0.275874</td>
      <td>0.171694</td>
      <td>0.278954</td>
      <td>0.410185</td>
      <td>0.442884</td>
      <td>0.361813</td>
      <td>0.371340</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>27.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>6.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>267.000000</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>9.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>313.000000</td>
      <td>6.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>4.000000</td>
      <td>7.000000</td>
      <td>26.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>903.000000</td>
      <td>6.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.00000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 55 columns</p>
</div>




```python
b.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>education</th>
      <th>ofjobs</th>
      <th>yearsexp</th>
      <th>honors</th>
      <th>volunteer</th>
      <th>military</th>
      <th>empholes</th>
      <th>occupspecific</th>
      <th>occupbroad</th>
      <th>workinschool</th>
      <th>...</th>
      <th>educreq</th>
      <th>compreq</th>
      <th>orgreq</th>
      <th>manuf</th>
      <th>transcom</th>
      <th>bankreal</th>
      <th>trade</th>
      <th>busservice</th>
      <th>othservice</th>
      <th>missind</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2435.000000</td>
      <td>2435.000000</td>
      <td>2435.000000</td>
      <td>2435.000000</td>
      <td>2435.000000</td>
      <td>2435.000000</td>
      <td>2435.000000</td>
      <td>2435.000000</td>
      <td>2435.000000</td>
      <td>2435.000000</td>
      <td>...</td>
      <td>2435.000000</td>
      <td>2435.000000</td>
      <td>2435.00000</td>
      <td>2435.000000</td>
      <td>2435.000000</td>
      <td>2435.000000</td>
      <td>2435.000000</td>
      <td>2435.000000</td>
      <td>2435.000000</td>
      <td>2435.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.616016</td>
      <td>3.658316</td>
      <td>7.829569</td>
      <td>0.051335</td>
      <td>0.414374</td>
      <td>0.101848</td>
      <td>0.445996</td>
      <td>216.744969</td>
      <td>3.487885</td>
      <td>0.560986</td>
      <td>...</td>
      <td>0.106776</td>
      <td>0.437372</td>
      <td>0.07269</td>
      <td>0.082957</td>
      <td>0.030390</td>
      <td>0.085010</td>
      <td>0.213963</td>
      <td>0.267762</td>
      <td>0.154825</td>
      <td>0.165092</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.733060</td>
      <td>1.219150</td>
      <td>5.010764</td>
      <td>0.220725</td>
      <td>0.492715</td>
      <td>0.302511</td>
      <td>0.497177</td>
      <td>148.021857</td>
      <td>2.043125</td>
      <td>0.496369</td>
      <td>...</td>
      <td>0.308892</td>
      <td>0.496164</td>
      <td>0.25968</td>
      <td>0.275874</td>
      <td>0.171694</td>
      <td>0.278954</td>
      <td>0.410185</td>
      <td>0.442884</td>
      <td>0.361813</td>
      <td>0.371340</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>27.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>6.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>267.000000</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>9.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>313.000000</td>
      <td>6.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>4.000000</td>
      <td>7.000000</td>
      <td>44.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>903.000000</td>
      <td>6.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.00000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 55 columns</p>
</div>



Apparently the problem is mischaracterized here. It states that "identical" resumes are assigned black or white sounding names, but the distributions of resume attributes are *not* identical for white and black resumes. They are certainly very close to each other, but they are not identical.

This problem is interesting, as it's just a binary classification problem with an imbalanced dataset. But we want to know if this imbalance is different for two separate subsets within the data. This is represented as a Bernoulli Distribution, in which you have just 0s and 1s, and the probability of getting a 0 or 1 is not (necessarily) 50%. The probability of getting a 1 is p, and the probability of getting a 0 is 1-p, defined as q. As such, the distribution means are given by the probability p. Let's calculate these means for both the black- and white-sounding names:


```python
# Calculate the means (p) for each distribution as a percentage:
mean_w = sum(w.call)/len(w.call)
mean_b = sum(b.call)/len(b.call)
print(mean_w*100., mean_b*100.)
```

    9.650924024640657 6.447638603696099


These are some small means, which makes sense given the imbalanced problem. To perform a z-test, we also need to calculate the variance of the Bernoulli Distributions here, which is just the product of both p and q probabilities:


```python
# Calculate the variances for each distribution (pq):
var_w = mean_w * (1-mean_w)
var_b = mean_b * (1-mean_b)
print(var_w, var_b)
```

    0.08719520679346794 0.060319181680573764



```python
# Calculate the z-statistic between the two distributions to test if they have the same mean:
import scipy.stats as st

N = len(w.call)   # Same N for both distributions

z_stat = (mean_b - mean_w) / np.sqrt((var_b + var_w) / N)
print('Z value = {}'.format(z_stat))

# Figure out the 95% confidence intervals:
print('95% confidence interval z-table range: ')
print([st.norm.ppf(.025), st.norm.ppf(.975)])

# Calculate the p-value:
p_val = st.norm.cdf(z_stat)
print('p value: {}'.format(p_val))
```

    Z value = -4.11555043573
    95% confidence interval z-table range: 
    [-1.9599639845400545, 1.959963984540054]
    p value: 1.931282603761311e-05


Using the z-test, we can very clearly reject the null hypothesis, meaning that there is definitely a statistical bias in the callback process regarding racial names. The question asks us to examine the question again using a bootstrap method, so that will be the next thing to do:


```python
def bootstrap_1d(data, func):
    """
    This will take a 1d array of data and create a bootstrap replicate
    of the supplied statistical function (mean, median, std, etc.).
    
    """
    
    return func(np.random.choice(data, size=len(data)))

def draw_bs_reps(data, func, size=1):
    """
    Draw bootstrap replicates of the supplied statistical function.
    Size here is how many total replicates you wish to draw.
    
    """

    # Initialize array of replicates to be filled:
    bs_replicates = np.empty(size)
    
    # Generate all the replicates and store them
    for i in range(size):
        bs_replicates[i] = bootstrap_1d(data, func)
    
    return bs_replicates

# Let's draw a sufficient number of bootstrap replicates of the mean, which is a sum for this distribution:
bs_reps = draw_bs_reps(w.call, np.sum, size=1000) / len(w.call)
rep_mean = np.mean(bs_reps)*100.0
print('Replicate mean (as a percentage): {}'.format(rep_mean))   # Check the mean of the replicates
```

    Replicate mean (as a percentage): 9.667967145790554



```python
# Plot out the replicate mean distribution compared to the mean of black-sounding names:
import matplotlib.pyplot as plt

weights_bs = np.ones_like(bs_reps)/float(len(bs_reps))
fig, axes = plt.subplots(figsize=(10,8))
_ = axes.hist(bs_reps, bins=8, weights=weights_bs)
_ = axes.plot(np.array([mean_b, mean_b]), np.array([0.0, 0.35]), color='black', alpha=0.5)
_ = axes.set_ylim([0.0, 0.35])
_ = axes.set_xlabel('Callback Means', fontsize=18)
_ = axes.set_ylabel('PDF', fontsize=18)
_ = plt.legend(['Mean of Black-Sounding Names', 'Replicate Means of White-Sounding Names'], fontsize=10)
_ = plt.title('Distribution of Potential Means for White-Sounding Names', fontsize=18)
plt.show()
```


![png](output_19_0.png)



```python
# Calculate p_val for rejecting null hypothesis:

unit_distance = rep_mean - mean_b
lower_vals = bs_reps[np.where(bs_reps <= (rep_mean - unit_distance))]
upper_vals = bs_reps[np.where(bs_reps >= (rep_mean + unit_distance))]
p_val = (np.sum(lower_vals) + np.sum(upper_vals)) / len(bs_reps)
print(p_val, len(lower_vals), len(upper_vals))
```

    0.0 0 0


As can be clearly seen from the plot, even with 1000 bootstrap replacement means, the callback mean for white-sounding names never gets close to being as low as the mean for black-sounding names. As such, the p-value is zero. We can therefore easily reject the null hypothesis, as we were able to do for the z-test.

<div class="span5 alert alert-success">
<p> Your answers to Q4 and Q5 here 

4. According to these results, it is clear that the probability for employers to call back a candidate drops significantly if the candidate has a black-sounding name compared to a white-sounding name. That means that race clearly plays a factor in the hiring process, even if not done intentionally.

5. The above answer notwithstanding, it cannot be said that race is the only factor at play in this analysis. There are a number of additional factors that can be at play. For example, were demographics of the hiring managers controlled for? How about their geographic location? I also showed in the beginning of this analysis that the resumes were not, in fact, identical, which means the resumes themselves were not a control factor. In order to determine if this is the most important factor, we would need to identify all of the relevant information, much of which can be found in the data (and likely some that isn't and would therefore need to be acquired). Once all the data is acquired and cleaned, a feature importance analysis would help figure out whether or not race is the most important factor. This could involve something like PCA, Regularization techniques (Lasso or Ridge Regression), or some modeling with interpretable models (Logistic Regression, Random Forest, Decision Trees, etc.).
    
</p>
</div>
