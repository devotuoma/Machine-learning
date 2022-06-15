```python
# Surport vector machine
import pandas as pd
import numpy as np 
from sklearn import svm  
# visual your data
import matplotlib.pyplot as plt
import seaborn as sns;sns.set(font_scale=1.2)
%matplotlib inline
```


```python
df = pd.read_csv('COVID-19_Vaccine.csv')
df
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
      <th>Website</th>
      <th>Article</th>
      <th>ArticleTitle</th>
      <th>ArticleText</th>
      <th>Author(s)</th>
      <th>ArticleLink</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CNN</td>
      <td>1</td>
      <td>Perfect storm' of disease ahead with vaccines ...</td>
      <td>The World Health Organization and the United N...</td>
      <td>Virginia Langmaid</td>
      <td>https://www.cnn.com/2022/04/27/health/who-unic...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CNN</td>
      <td>2</td>
      <td>Pfizer requests FDA authorization for Covid-19...</td>
      <td>Pfizer and BioNTech said Tuesday that they hav...</td>
      <td>Jen Christensen</td>
      <td>https://www.cnn.com/2022/04/26/health/pfizer-b...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CNN</td>
      <td>3</td>
      <td>FDA approves remdesivir to treat young childre...</td>
      <td>The US Food and Drug Administration announced ...</td>
      <td>Jamie Gumbrecht, Jacquelin Howard</td>
      <td>https://www.cnn.com/2022/04/25/health/fda-remd...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CNN</td>
      <td>4</td>
      <td>How well is our immunity holding up against Co...</td>
      <td>Now that most US cities and states have droppe...</td>
      <td>Brenda Goodman</td>
      <td>https://www.cnn.com/2022/04/22/health/immunity...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CNN</td>
      <td>5</td>
      <td>When will the US have a Covid-19 vaccine for t...</td>
      <td>It's been more than a year since adults first ...</td>
      <td>Jen Christensen</td>
      <td>https://www.cnn.com/2022/04/22/health/vaccine-...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Fox</td>
      <td>1</td>
      <td>FDA panel endorses Johnson &amp; Johnson COVID-19 ...</td>
      <td>A Food and Drug Administration (FDA) advisory ...</td>
      <td>Kayla Rivas</td>
      <td>https://www.foxnews.com/health/fda-panel-johns...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Fox</td>
      <td>2</td>
      <td>FDA panel endorses Pfizer’s COVID-19 vaccine b...</td>
      <td>A U.S. Food and Drug Administration (FDA) advi...</td>
      <td>Kayla Rivas</td>
      <td>https://www.foxnews.com/health/fda-panel-pfize...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Fox</td>
      <td>3</td>
      <td>De Blasio to pay kids $100 to get the COVID-19...</td>
      <td>New York City will begin offering $100 to ince...</td>
      <td>Michael Lee</td>
      <td>https://www.foxnews.com/politics/de-blasio-pay...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Fox</td>
      <td>4</td>
      <td>COVID-19 vaccine booster and flu shot: Is it s...</td>
      <td>As the flu season approaches and coincides wit...</td>
      <td>Amy McGorry</td>
      <td>https://www.foxnews.com/health/covid-vaccine-b...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Fox</td>
      <td>5</td>
      <td>Will Johnson &amp; Johnson COVID-19 vaccine recipi...</td>
      <td>The plan for booster shots laid out by health ...</td>
      <td>Alexandria Hein</td>
      <td>https://www.foxnews.com/health/johnson-johnson...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>ABC</td>
      <td>1</td>
      <td>Measles outbreaks possible amid 'perfect storm...</td>
      <td>The World Health Organization and UNICEF are w...</td>
      <td>Sasha Pezenik</td>
      <td>https://abcnews.go.com/Health/measles-outbreak...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>ABC</td>
      <td>2</td>
      <td>Fauci said the US is 'out of the pandemic phas...</td>
      <td>The worst days of the COVID-19 pandemic may be...</td>
      <td>Mary Kekatos</td>
      <td>https://abcnews.go.com/Health/fauci-us-pandemi...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>ABC</td>
      <td>3</td>
      <td>Parents of kids under 5 fed up with lack of FD...</td>
      <td>Parents eager to vaccinate their toddlers and ...</td>
      <td>Anne Flaherty</td>
      <td>https://abcnews.go.com/Politics/parents-kids-f...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>ABC</td>
      <td>4</td>
      <td>Millions of COVID-19 shots set to go to waste,...</td>
      <td>While top U.S. health officials are urging som...</td>
      <td>Arielle Mitropoulos</td>
      <td>https://abcnews.go.com/Health/millions-covid-1...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>ABC</td>
      <td>5</td>
      <td>More than 100 million Americans have received ...</td>
      <td>More than 100 million Americans have received ...</td>
      <td>Mary Kekatos</td>
      <td>https://abcnews.go.com/Health/100-million-amer...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>MSNBC</td>
      <td>1</td>
      <td>How this 'little ole girl' from North Carolina...</td>
      <td>She’s a globally renowned scientist and Covid-...</td>
      <td>Donna M. Owens</td>
      <td>https://www.msnbc.com/know-your-value/career-g...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>MSNBC</td>
      <td>2</td>
      <td>The Covid-19 vaccine and PrEP both require tal...</td>
      <td>After over a year of living in fear, the Cente...</td>
      <td>Zach Stafford</td>
      <td>https://www.msnbc.com/opinion/covid-19-vaccine...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>MSNBC</td>
      <td>3</td>
      <td>The world is counting on us': How Pfizer's hea...</td>
      <td>As vice president and head of clinical trial e...</td>
      <td>Know Your Value staff</td>
      <td>https://www.msnbc.com/know-your-value/business...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>MSNBC</td>
      <td>4</td>
      <td>Covid-19 vaccine booster shots might be needed...</td>
      <td>On a recent conference call with medical exper...</td>
      <td>Dr. Kavita Patel</td>
      <td>https://www.msnbc.com/opinion/covid-19-vaccine...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>MSNBC</td>
      <td>5</td>
      <td>U.S. Covid-19 'vaccine diplomacy' is catching ...</td>
      <td>What a difference a few months can make. After...</td>
      <td>Hayes Brown</td>
      <td>https://www.msnbc.com/opinion/u-s-covid-19-vac...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>BBC</td>
      <td>1</td>
      <td>Covid: Job losses expected at Wrexham vaccine ...</td>
      <td>Jobs losses are expected to be announced at a ...</td>
      <td>Unknown</td>
      <td>https://www.bbc.com/news/uk-wales-61222595</td>
    </tr>
    <tr>
      <th>21</th>
      <td>BBC</td>
      <td>2</td>
      <td>Covid: Violence up after lockdown eased and jo...</td>
      <td>Here are five things you need to know about th...</td>
      <td>Unknown</td>
      <td>https://www.bbc.com/news/uk-61216740</td>
    </tr>
    <tr>
      <th>22</th>
      <td>BBC</td>
      <td>3</td>
      <td>Covid vaccines: How fast is progress around th...</td>
      <td>More than 11.5 billion doses of coronavirus va...</td>
      <td>Visual and Data Journalism Team</td>
      <td>https://www.bbc.com/news/world-56237778</td>
    </tr>
    <tr>
      <th>23</th>
      <td>BBC</td>
      <td>4</td>
      <td>Covaxin: India approves two Covid vaccines for...</td>
      <td>India has approved two homegrown vaccines for ...</td>
      <td>Unknown</td>
      <td>https://www.bbc.com/news/world-asia-india-5574...</td>
    </tr>
    <tr>
      <th>24</th>
      <td>BBC</td>
      <td>5</td>
      <td>Covid-19 vaccine offered to children aged five...</td>
      <td>All children aged between five and 11 in North...</td>
      <td>Unknown</td>
      <td>https://www.bbc.com/news/uk-northern-ireland-6...</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Guardian</td>
      <td>1</td>
      <td>How the race for a Covid-19 vaccine is getting...</td>
      <td>To begin with, it felt like a sleek performanc...</td>
      <td>Laura Spinney</td>
      <td>https://www.theguardian.com/society/2020/aug/3...</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Guardian</td>
      <td>2</td>
      <td>Covid-19 vaccines: the contracts, prices and p...</td>
      <td>Two US companies, Pfizer and Moderna, have rai...</td>
      <td>Julia Kollewe</td>
      <td>https://www.theguardian.com/world/2021/aug/11/...</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Guardian</td>
      <td>3</td>
      <td>Texas scientists’ new Covid-19 vaccine is chea...</td>
      <td>A new Covid-19 vaccine is being developed by T...</td>
      <td>Erum Salam</td>
      <td>https://www.theguardian.com/us-news/2022/jan/1...</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Guardian</td>
      <td>4</td>
      <td>Patient removed from heart transplant list for...</td>
      <td>A Boston-area hospital said it will not perfor...</td>
      <td>Gloria Oladipo</td>
      <td>https://www.theguardian.com/us-news/2022/jan/2...</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Guardian</td>
      <td>5</td>
      <td>World leaders pledge €7.4bn to research Covid-...</td>
      <td>World leaders, with the notable exception of D...</td>
      <td>Patrick Wintour</td>
      <td>https://www.theguardian.com/world/2020/may/04/...</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.describe()
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
      <th>Article</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>30.00000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.00000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.43839</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.00000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.00000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.00000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.00000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>5.00000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.tail()
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
      <th>Website</th>
      <th>Article</th>
      <th>ArticleTitle</th>
      <th>ArticleText</th>
      <th>Author(s)</th>
      <th>ArticleLink</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>25</th>
      <td>Guardian</td>
      <td>1</td>
      <td>How the race for a Covid-19 vaccine is getting...</td>
      <td>To begin with, it felt like a sleek performanc...</td>
      <td>Laura Spinney</td>
      <td>https://www.theguardian.com/society/2020/aug/3...</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Guardian</td>
      <td>2</td>
      <td>Covid-19 vaccines: the contracts, prices and p...</td>
      <td>Two US companies, Pfizer and Moderna, have rai...</td>
      <td>Julia Kollewe</td>
      <td>https://www.theguardian.com/world/2021/aug/11/...</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Guardian</td>
      <td>3</td>
      <td>Texas scientists’ new Covid-19 vaccine is chea...</td>
      <td>A new Covid-19 vaccine is being developed by T...</td>
      <td>Erum Salam</td>
      <td>https://www.theguardian.com/us-news/2022/jan/1...</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Guardian</td>
      <td>4</td>
      <td>Patient removed from heart transplant list for...</td>
      <td>A Boston-area hospital said it will not perfor...</td>
      <td>Gloria Oladipo</td>
      <td>https://www.theguardian.com/us-news/2022/jan/2...</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Guardian</td>
      <td>5</td>
      <td>World leaders pledge €7.4bn to research Covid-...</td>
      <td>World leaders, with the notable exception of D...</td>
      <td>Patrick Wintour</td>
      <td>https://www.theguardian.com/world/2020/may/04/...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Handling Categorical Data
# Creating dummy variables
import pandas as pd
from patsy import dmatrices
df = pd.DataFrame({'A': ['high', 'medium', 'low'],
                  'B': [10,20,30]},
                 index=[0,1,2])
df
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
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>high</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>medium</td>
      <td>20</td>
    </tr>
    <tr>
      <th>2</th>
      <td>low</td>
      <td>30</td>
    </tr>
  </tbody>
</table>
</div>




```python
# using get_dummies function of pandas package
df_with_dummies= pd.get_dummies(df, prefix='A', columns=['A'])
print (df_with_dummies)
```

        B  A_high  A_low  A_medium
    0  10       1      0         0
    1  20       0      0         1
    2  30       0      1         0



```python
import pandas as pd
```

# Surpervised Learning


```python
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
# LOad data
df = pd.read_csv('squark_automotive_CLV_production_data.csv')
print(df)
```

         Customer       State Response  Coverage Education Effective To Date  \
    0     FH77504  California      Yes   Premium  Bachelor         1/24/2011   
    1     XK87182      Oregon       No   Premium   College         1/25/2011   
    2     HB17438  Washington       No  Extended  Bachelor         2/21/2011   
    3     DH18269     Arizona       No  Extended   College         1/13/2011   
    4     DP19820      Oregon       No  Extended   College         1/15/2011   
    ...       ...         ...      ...       ...       ...               ...   
    1030  LA72316  California       No     Basic  Bachelor         2/10/2011   
    1031  PK87824  California      Yes  Extended   College         2/12/2011   
    1032  TD14365  California       No  Extended  Bachelor          2/6/2011   
    1033  UP19263  California       No  Extended   College          2/3/2011   
    1034  Y167826  California       No  Extended   College         2/14/2011   
    
         EmploymentStatus Gender  Income Location Code  ...  \
    0            Employed      F   51643      Suburban  ...   
    1            Employed      F   46402         Urban  ...   
    2            Employed      M   92044         Urban  ...   
    3       Medical Leave      M   16040      Suburban  ...   
    4          Unemployed      M       0      Suburban  ...   
    ...               ...    ...     ...           ...  ...   
    1030         Employed      M   71941         Urban  ...   
    1031         Employed      F   21604      Suburban  ...   
    1032       Unemployed      M       0      Suburban  ...   
    1033         Employed      M   21941      Suburban  ...   
    1034       Unemployed      M       0      Suburban  ...   
    
         Months Since Policy Inception  Number of Open Complaints  \
    0                               43                          0   
    1                                2                          0   
    2                               77                          0   
    3                               93                          0   
    4                               84                          4   
    ...                            ...                        ...   
    1030                            89                          0   
    1031                            28                          0   
    1032                            37                          3   
    1033                             3                          0   
    1034                            90                          0   
    
          Number of Policies     Policy Type        Policy  Renew Offer Type  \
    0                      1   Personal Auto   Personal L3            Offer2   
    1                      1   Personal Auto   Personal L3            Offer1   
    2                      3   Personal Auto   Personal L1            Offer4   
    3                      2   Personal Auto   Personal L3            Offer1   
    4                      2   Personal Auto   Personal L2            Offer2   
    ...                  ...             ...           ...               ...   
    1030                   2   Personal Auto   Personal L1            Offer2   
    1031                   1  Corporate Auto  Corporate L3            Offer1   
    1032                   2  Corporate Auto  Corporate L2            Offer1   
    1033                   3   Personal Auto   Personal L2            Offer3   
    1034                   1  Corporate Auto  Corporate L3            Offer4   
    
         Sales Channel Total Claim Amount  Vehicle Class Vehicle Size  
    0            Agent        1358.400000     Luxury Car      Medsize  
    1            Agent         476.385575  Four-Door Car      Medsize  
    2              Web         617.288574            SUV      Medsize  
    3           Branch         611.476898   Two-Door Car      Medsize  
    4           Branch         980.528170            SUV        Small  
    ...            ...                ...            ...          ...  
    1030           Web         198.234764  Four-Door Car      Medsize  
    1031        Branch         379.200000  Four-Door Car      Medsize  
    1032        Branch         790.784983  Four-Door Car      Medsize  
    1033        Branch         691.200000  Four-Door Car        Large  
    1034   Call Center         369.600000   Two-Door Car      Medsize  
    
    [1035 rows x 23 columns]



```python
df.describe()
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
      <th>Income</th>
      <th>Monthly Premium Auto</th>
      <th>Months Since Last Claim</th>
      <th>Months Since Policy Inception</th>
      <th>Number of Open Complaints</th>
      <th>Number of Policies</th>
      <th>Total Claim Amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1035.000000</td>
      <td>1035.000000</td>
      <td>1035.000000</td>
      <td>1035.000000</td>
      <td>1035.000000</td>
      <td>1035.000000</td>
      <td>1035.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>36200.622222</td>
      <td>93.833816</td>
      <td>15.296618</td>
      <td>47.415459</td>
      <td>0.346860</td>
      <td>2.984541</td>
      <td>454.624993</td>
    </tr>
    <tr>
      <th>std</th>
      <td>30428.327030</td>
      <td>34.627633</td>
      <td>9.970359</td>
      <td>28.584926</td>
      <td>0.879861</td>
      <td>2.389887</td>
      <td>318.034652</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>61.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2.823303</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>69.000000</td>
      <td>6.000000</td>
      <td>22.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>290.680278</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>30064.000000</td>
      <td>85.000000</td>
      <td>14.000000</td>
      <td>47.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>390.498822</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>60926.000000</td>
      <td>110.000000</td>
      <td>24.000000</td>
      <td>72.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>561.600000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>99845.000000</td>
      <td>286.000000</td>
      <td>35.000000</td>
      <td>99.000000</td>
      <td>5.000000</td>
      <td>9.000000</td>
      <td>2452.894264</td>
    </tr>
  </tbody>
</table>
</div>




```python
# check the correlation between variables
print (df. corr())
```

                                     Income  Monthly Premium Auto  \
    Income                         1.000000             -0.051720   
    Monthly Premium Auto          -0.051720              1.000000   
    Months Since Last Claim       -0.038026             -0.001538   
    Months Since Policy Inception  0.007695             -0.028212   
    Number of Open Complaints     -0.037015              0.032303   
    Number of Policies             0.009107             -0.035230   
    Total Claim Amount            -0.381239              0.671513   
    
                                   Months Since Last Claim  \
    Income                                       -0.038026   
    Monthly Premium Auto                         -0.001538   
    Months Since Last Claim                       1.000000   
    Months Since Policy Inception                -0.073373   
    Number of Open Complaints                     0.006120   
    Number of Policies                            0.003602   
    Total Claim Amount                            0.002222   
    
                                   Months Since Policy Inception  \
    Income                                              0.007695   
    Monthly Premium Auto                               -0.028212   
    Months Since Last Claim                            -0.073373   
    Months Since Policy Inception                       1.000000   
    Number of Open Complaints                          -0.018886   
    Number of Policies                                  0.016332   
    Total Claim Amount                                 -0.046341   
    
                                   Number of Open Complaints  Number of Policies  \
    Income                                         -0.037015            0.009107   
    Monthly Premium Auto                            0.032303           -0.035230   
    Months Since Last Claim                         0.006120            0.003602   
    Months Since Policy Inception                  -0.018886            0.016332   
    Number of Open Complaints                       1.000000            0.024629   
    Number of Policies                              0.024629            1.000000   
    Total Claim Amount                              0.065619            0.014168   
    
                                   Total Claim Amount  
    Income                                  -0.381239  
    Monthly Premium Auto                     0.671513  
    Months Since Last Claim                  0.002222  
    Months Since Policy Inception           -0.046341  
    Number of Open Complaints                0.065619  
    Number of Policies                       0.014168  
    Total Claim Amount                       1.000000  



```python
#Simple scatter plot
df.plot(kind='scatter', x='Income', y='Total Claim Amount', title='Income vs Total Claim Amount')
```




    <AxesSubplot:title={'center':'Income vs Total Claim Amount'}, xlabel='Income', ylabel='Total Claim Amount'>




    
![png](output_12_1.png)
    



```python

```
