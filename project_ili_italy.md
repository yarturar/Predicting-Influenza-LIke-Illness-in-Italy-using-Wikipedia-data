
# Part 1

# 1.1:


In this section, we are going to get access to data provided by Wikimedia, specifically, number of pageviews for "Influenza" page. Using their API, we can get the pageview number as json file, which should be processed further. Processing this data includes aggregating pageview count to week scale, since data is provided in day scale. Considering limitation of data, this tool of Wikimedia had been started from July of 2015, therefore, we will take data from that time until now.  


```python
import json
import datetime   #library for converting date to week number of a year
from ast import literal_eval 
import pandas as pd
import urllib
import matplotlib.pyplot as plt
```


```python
main='https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/' #main part of url 
platform ='it.wikipedia/'                        #here we selected wikipedia platform for Italy
access='all-access/'                             #all access including mobile, desktop and tablet
who='user/'                                      #we are interested in users (not spider or bots)
pagename='Influenza/'                            #name of page
date='daily/20150601/20181202'                   #date and scale for data


url = main+platform+access+who+pagename+date
response = urllib.request.urlopen(url)
data = response.read()          # a `bytes` object
text = data.decode('utf-8')     #converting from bytes object into string
tup = literal_eval(text)        #unstring the object
```


```python
#this function takes dictionary data, and then returns dataframe containing pageview numbers in year-week scale  

def year_week_pageview(tup):    
    table=pd.DataFrame(columns=["Year", "Week", "Pageview"])    #creating empty dataframe with the columns
    for i in range(len(tup['items'])):
        year=int(tup['items'][i]['timestamp'][:4])
        month=int(tup['items'][i]['timestamp'][4:6])
        day=int(tup['items'][i]['timestamp'][6:8])
        date=datetime.date(year, month, day).isocalendar()      #getting year and week number for current date
        table=table.append({"Year": date[0], "Week": date[1], "Pageview": int(tup['items'][i]['views'])},
                          ignore_index=True)
    grouped=table.groupby(['Year','Week'])['Pageview'].sum().reset_index(name="Pageview")   #grouping by year and then by week
    return(grouped)
```


```python
df=year_week_pageview(tup)
df.head(5)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Week</th>
      <th>Pageview</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015</td>
      <td>27</td>
      <td>493</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015</td>
      <td>28</td>
      <td>642</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015</td>
      <td>29</td>
      <td>646</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015</td>
      <td>30</td>
      <td>584</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015</td>
      <td>31</td>
      <td>554</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape    #overall 179 weeks 
```




    (179, 3)




```python
#here we will plot Pageview number for "Influenza" versus year-week 


index=[]
index=df['Year'].apply(str)+"-"+df['Week'].apply(str)  #creating index for x axis 


ax=df.plot(x=index, y='Pageview', title="Wikipedia Pageview data for 'Influenza'",rot=90, figsize=(10,5), legend=False,
            kind="line", xticks=range(0, 179, 5))   #plot, and x axis ticks for every 5 year-week index


ax.set_xlabel("Year-Week")
ax.set_ylabel("Number of Pageviews")
ax.grid( linewidth='0.2', color='grey')
plt.show()
```


![png](output_8_0.png)


# 1.2:


In this part, we will extract tables from Influnet pdf files using Tabula, and compare them against "Influenza" Wikipedia data in terms of correlation and fitness. Influnet shows data from 1st to 17th week and from 42nd to 52/53 th week for each year. That is why, our Wikipedia data should be mached in these periods.


```python
import numpy
import sklearn.metrics
```


```python
#after extracting tables from the pdfs, we can access them by csv reader of python

data_15_16=pd.read_csv(r'C:path\tabula-InfluNet - Stagione 2015 - 2016.csv',
                sep=",")
data_16_17=pd.read_csv(r'C:path\tabula-InfluNet - Stagione 2016 - 2017.csv',
                sep=",")
data_17_18=pd.read_csv(r'C:path\tabula-InfluNet - Stagione 2017 - 2018.csv',
                sep=",")

data_15_16.head(5)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Totale</th>
      <th>Totale.1</th>
      <th>Totale.2</th>
      <th>Incidenza</th>
      <th>0 - 4</th>
      <th>5 - 14</th>
      <th>15 - 64</th>
      <th>65 e oltre</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Settimana</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>Medici</td>
      <td>Casi</td>
      <td>Assistiti</td>
      <td>Totale</td>
      <td>Casi Inc</td>
      <td>Casi Inc</td>
      <td>Casi Inc</td>
      <td>Casi Inc</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015-42</td>
      <td>821</td>
      <td>425</td>
      <td>1.065.225</td>
      <td>0,40</td>
      <td>75 1,11</td>
      <td>60 0,42</td>
      <td>228 0,37</td>
      <td>62 0,26</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015-43</td>
      <td>928</td>
      <td>604</td>
      <td>1.204.096</td>
      <td>0,50</td>
      <td>112 1,45</td>
      <td>82 0,51</td>
      <td>330 0,47</td>
      <td>80 0,30</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015-44</td>
      <td>956</td>
      <td>842</td>
      <td>1.245.366</td>
      <td>0,68</td>
      <td>156 1,91</td>
      <td>108 0,64</td>
      <td>469 0,65</td>
      <td>109 0,40</td>
    </tr>
  </tbody>
</table>
</div>



As we can see, we need to take values from the second row for the 1st column and 5th column, namely "Unnamed: 0" and "Incidenza". Here, Incidenza shows number of cases per 1000 people.


```python
#filtering data and merging them together
d1=data_15_16[["Unnamed: 0", "Incidenza"]].iloc[2:]
d2=data_16_17[["Unnamed: 0", "Incidenza"]].iloc[2:]
d3=data_17_18[["Unnamed: 0", "Incidenza"]].iloc[2:]
merged=pd.concat([d1, d2,d3], ignore_index=True)
merged.head(5)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Incidenza</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015-42</td>
      <td>0,40</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015-43</td>
      <td>0,50</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015-44</td>
      <td>0,68</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015-45</td>
      <td>0,76</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015-46</td>
      <td>0,88</td>
    </tr>
  </tbody>
</table>
</div>



At this point, we should modify this table. First of all,we will take out values from column "Unnamed:0" and write them sperately as Year and Week for new columns "Year" and "Week" respectively. Secondly, we need change format of "Incidenza" column values (ex. from '0,4' to 0.4), that is, replace commas with dots and convert from string to float type


```python

influnet=pd.DataFrame()   #create new dataframe

influnet['Year'], influnet['Week'] = merged['Unnamed: 0'].str.split('-', 1).str    #split year and week
influnet['Influnet']=merged['Incidenza']   
influnet['Influnet']=influnet['Influnet'].apply(lambda x: float(x.replace(',', '.'))) #replace commas with dots
influnet['Year']=influnet['Year'].apply(int)    #from str to int
influnet['Week']=influnet['Week'].apply(int)

influnet.head(5)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Week</th>
      <th>Influnet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015</td>
      <td>42</td>
      <td>0.40</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015</td>
      <td>43</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015</td>
      <td>44</td>
      <td>0.68</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015</td>
      <td>45</td>
      <td>0.76</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015</td>
      <td>46</td>
      <td>0.88</td>
    </tr>
  </tbody>
</table>
</div>



Now, we will join "Influenza" dataframe from the previous part into this dataframe on columns Year and Week


```python
comb=pd.merge(df, influnet, on=["Year","Week"])
comb.head(5)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Week</th>
      <th>Pageview</th>
      <th>Influnet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015</td>
      <td>42</td>
      <td>1400</td>
      <td>0.40</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015</td>
      <td>43</td>
      <td>1317</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015</td>
      <td>44</td>
      <td>1226</td>
      <td>0.68</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015</td>
      <td>45</td>
      <td>1336</td>
      <td>0.76</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015</td>
      <td>46</td>
      <td>1298</td>
      <td>0.88</td>
    </tr>
  </tbody>
</table>
</div>




```python
comb.shape
```




    (84, 4)



Combined table has 84 weeks


```python
index=[]

index=comb['Year'].apply(str)+"-"+comb['Week'].apply(str)
ax=comb.plot(x=index, y='Pageview', title=" 'Influenza' Wiki vs Influnet",rot=90, figsize=(10,5), legend=False,
            kind="line", xticks=range(0, 84, 5))

ax2 = ax.twinx()
comb.plot(x=index, y=['Influnet'],ax=ax2, color='r',rot=90, figsize=(10,5), legend=True,
            kind="line", xticks=range(0, 84, 5))
ax2.set_ylabel("Case per 1000 people - Influnet")
ax2.legend(bbox_to_anchor=(-0.85, 0.8, 1., .102))
ax.legend(labels=('Influenza Wiki',))
ax.set_xlabel("Year-Week")
ax.set_ylabel("Number of Pageviews - Wiki")
ax.grid( linewidth='0.2', color='grey')
plt.show()
```


![png](output_21_0.png)


Here, we will calculate Pearson correlation coefficient for Influenza and Influnet data for different lags of time. That is, we will shift time scale for Wikipedia data, calling changing lags, to see how correlations change. For example, Wiki-1 means that we will correlate Wikipedia Data for week n-1 with Influnet data in week n. 


```python
lag_corr=pd.DataFrame(columns=['Wiki','Wiki-1','Wiki-2','Wiki+1','Wiki+2']) #creating empty table 

#pearson correlation coeficcient for different lags
lag_corr=lag_corr.append({'Wiki':round( numpy.corrcoef(comb["Pageview"], comb["Influnet"])[0, 1],3),
                         'Wiki-1':round(numpy.corrcoef(comb["Pageview"].iloc[:83], comb["Influnet"].iloc[1:])[0, 1],3), 
                         'Wiki-2':round(numpy.corrcoef(comb["Pageview"].iloc[:82], comb["Influnet"].iloc[2:])[0, 1],3),
                         'Wiki+1':round(numpy.corrcoef(comb["Pageview"].iloc[1:], comb["Influnet"].iloc[:83])[0, 1],3),
                         'Wiki+2':round(numpy.corrcoef(comb["Pageview"].iloc[2:], comb["Influnet"].iloc[:82])[0, 1],3)},
                                        ignore_index=True)

lag_corr.rename(index={0: 'Influnet'}, inplace=True)
print("Correlation for different lags:")
lag_corr

```

    Correlation for different lags:
    




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Wiki</th>
      <th>Wiki-1</th>
      <th>Wiki-2</th>
      <th>Wiki+1</th>
      <th>Wiki+2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Influnet</th>
      <td>0.918</td>
      <td>0.897</td>
      <td>0.805</td>
      <td>0.847</td>
      <td>0.688</td>
    </tr>
  </tbody>
</table>
</div>



Discussion: from figure above we can say that 'Influenza' wiki data and Influnet data have similar behaviour over time, that is, trends in Influnet coincides with trends in 'Influenza' wiki data. In order to see how they correlate, we calculated Pearson correlation coefficient for different time lags between them. Pearson correlation coefficient was chosen based on the provided article "mciver2014_fluwikipedia" in which authors used it. As a result, when there is not shifts in weeks between them, we observed highest value which is equal to 0.918. The second highest value was found for -1 week shift for wiki data, specifically, when wiki data in week n-1 were compared with Influnet data in week n. However, correlation for time series data should be considered carefully, because it could rise wrong assumptions. Pearson correlation coefficient can accurately show result for linear relationship, but in our case we have non-monotonic relationship. Nevertheless, Pearson correlation coefficient can be used in cases when we have stationary time series data with no significant trend. In our case time series data can be considered as weak stationary. So, in order to strenghten our assumptions about correlation, we can propose R square method, which will define how both data fitted to each other. Main idea behind this method is calculation of sum of squared errors for each pair of Wiki and Influnet data. Maximum value of R square could be 1, which indicates to perfect fitness of two data. Before conducting R square analysis, we need to normalize both data, because wiki data has value between 1000 and 6000, and range of Influnet is between 0 and 15. So, to normalize them in scale between 0 and 1, we can use following equation:
                    
                                        norm_x=(x-min_x)/(max_x-min_x)
                    


```python
#normalizing data into scale from 0 to 1
normalized=pd.DataFrame()
normalized['Year']=comb['Year']
normalized['Week']=comb['Week']
normalized["influenza_norm"]=(comb['Pageview']-min(comb["Pageview"]))/(max(comb["Pageview"])-min(comb["Pageview"]))
normalized["influnet_norm"]=(comb["Influnet"]-min(comb["Influnet"]))/(max(comb["Influnet"])-min(comb["Influnet"]))
```


```python
print('Normalized data:')
normalized.head(5)
```

    Normalized data:
    




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Week</th>
      <th>influenza_norm</th>
      <th>influnet_norm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015</td>
      <td>42</td>
      <td>0.089033</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015</td>
      <td>43</td>
      <td>0.072238</td>
      <td>0.006974</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015</td>
      <td>44</td>
      <td>0.053824</td>
      <td>0.019526</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015</td>
      <td>45</td>
      <td>0.076083</td>
      <td>0.025105</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015</td>
      <td>46</td>
      <td>0.068393</td>
      <td>0.033473</td>
    </tr>
  </tbody>
</table>
</div>



Lets plot and compare normalized Influenza wiki and Influnet data


```python
index=[]
index=normalized['Year'].apply(str)+"-"+normalized['Week'].apply(str)

ax=normalized.plot(x=index, y=['influenza_norm','influnet_norm'], title=" Normalized 'Influenza' Wiki vs Influnet",rot=90, figsize=(10,5), legend=True,
            kind="line", xticks=range(0, 84, 5))



ax.set_xlabel("Year-Week")
ax.set_ylabel("Number of Pageviews - Wiki")
ax.grid( linewidth='0.2', color='grey')
plt.show()
```


![png](output_28_0.png)


As we can see, this shows exactly the same plot as the previous one. Now, when we have data in the same scale we can calculate R square to evaluate fitness to each other. 


```python
import sklearn.metrics
sklearn.metrics.r2_score(normalized['influenza_norm'], normalized['influnet_norm'])
```




    0.791984435074526




```python
lag_r2=pd.DataFrame(columns=['Wiki','Wiki-1','Wiki-2','Wiki+1','Wiki+2']) #creating empty table 

#pearson correlation coeficcient for different lags
lag_r2=lag_r2.append({'Wiki':round( sklearn.metrics.r2_score(normalized['influenza_norm'], normalized['influnet_norm']),3),
                         'Wiki-1':round(sklearn.metrics.r2_score(normalized['influenza_norm'].iloc[:83], normalized['influnet_norm'].iloc[1:]),3), 
                         'Wiki-2':round(sklearn.metrics.r2_score(normalized['influenza_norm'].iloc[:82], normalized['influnet_norm'].iloc[2:]),3),
                         'Wiki+1':round(sklearn.metrics.r2_score(normalized['influenza_norm'].iloc[1:], normalized['influnet_norm'].iloc[:83]),3),
                         'Wiki+2':round(sklearn.metrics.r2_score(normalized['influenza_norm'].iloc[2:], normalized['influnet_norm'].iloc[:82]),3)},
                                        ignore_index=True)

lag_r2.rename(index={0: 'Influnet'}, inplace=True)
print("R square for different lags:")
lag_r2
```

    R square for different lags:
    




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Wiki</th>
      <th>Wiki-1</th>
      <th>Wiki-2</th>
      <th>Wiki+1</th>
      <th>Wiki+2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Influnet</th>
      <td>0.792</td>
      <td>0.744</td>
      <td>0.536</td>
      <td>0.631</td>
      <td>0.271</td>
    </tr>
  </tbody>
</table>
</div>



Table above shows when lag is zero between data, there is high fitness between them. Also, there is good fitness when lag is -1, that is, when wiki data in a week compared with influnet data of the next week. In conclusion, there is correlation between these data, and for further considerations in correlation we will consider lag=0 no shift between them

# Part 2

# 2.1

In this part we have to choose wikipedia pages, which correlate with Influnet data. First of all, I looked at symptoms of influenza such as temperature, rhinorrhoea, mialgia, cefalea, vomito and tosse. Secondly, I choose a page about virus itself called 'Orthomyxoviridae'. This is a family of RNA viruses that includes different type of Influenza viruses. Lastly, I picked up pages devoted for treatment of the virus. Here, the most important one is the page "Paracetamolo". Furthermore, if we search in Google for "trattamento dell'influenza", articles in the top results state that medicine with brand name 'Tamiflu','Ibuprofene' and 'Zanamivir' could be effective in treating the influenza. Therefore, I looked for their wiki pages and found that for 'Tamiflu' we have medical name called "Oseltamivir" on wikipedia, and for others it is the same as their names. For the last page, I selected 'Vaccino antinfluenzale', which is about vaccination to prevent yourself from influenza. So, I have 12 pages in total. 

Let's create a function that will take as input Wiki page names, and return dataframe with pagenumber in weekly scale for each page and influnet data


```python
def pages(page_list, influnet_data):
    main=influnet_data
    ull='https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/it.wikipedia/all-access/user/'
    date='/daily/20150601/20181202'    
    for page in page_list:
        url=ull+page+date
        response = urllib.request.urlopen(url)
        data = response.read()        
        text = data.decode('utf-8')     
        tup = literal_eval(text)
        page_data=year_week_pageview(tup)
        #since it returns column "Pageview", we have to change to corresponding value
        page_data.rename(columns={'Pageview':page}, inplace=True)  
        main=pd.merge(main, page_data, on=["Year","Week"])
    #main=pd.merge()
    return(main)
```


```python
from IPython.display import display, HTML   #this for display of dataframes
```


```python
page_list=['Febbre', 'Paracetamolo', 'Rinorrea', 'Orthomyxoviridae', 'Oseltamivir', 'Zanamivir', 'Ibuprofene',
          "Cefalea",'Mialgia', 'Vomito','Tosse', 'Vaccino antinfluenzale' ]  #page names


corr=pd.DataFrame(index=['Influnet'],columns=page_list)     #table for correlation
r2=pd.DataFrame(index=['Influnet'],columns=page_list)       #table for r square
v=pages(page_list, influnet)
page_list.append('Influnet')
var_list=page_list
w=v.copy()

```

At this point, lets draw for each page their time series behaviour in weekly base comparing with Influnet data.


```python
fig, axes = plt.subplots(nrows=4, ncols=3)   #for total 12 plots, I create subplot with shape(rows=4, colms=3)
for i, ax in enumerate(axes.reshape(-1)):
     v.plot(x=index, y=page_list[i],ax=ax, title=page_list[i]+" vs Influnet",rot=90, figsize=(20,15), legend=True)
     ax2 = ax.twinx()        #stacking Influnet data into a page's plot
     v.plot(x=index, y='Influnet',ax=ax2, color='r',rot=90, figsize=(20,15), legend=True,
                kind="line", xticks=range(0, 84, 5))
    
     ax2.legend(bbox_to_anchor=(-0.78, 0.8, 1., .102))
     
     ax.set_xlabel("Year-Week")
     ax.legend(loc='upper left')
     ax.grid( linewidth='0.2', color='grey')
        
     #here to place y and x  axis names properly, I only put them on the first and last plots
     if (i in [0,3,6,9]):
        ax.set_ylabel("Number of Pageviews - Wiki")
     elif (i in [2,5,8,11]): 
        ax2.set_ylabel("Case per 1000 people - Influnet")
        
plt.tight_layout()
plt.show()
```


![png](output_40_0.png)


From plots we can see that Febbre, Paracetamolo, Oseltamivir and Zanamivir show good correlation. Mialgia, Cefalea and Vomito have similar trend over time. These symptoms are also symptoms of other illnesses, and therefore they could show different trend than influenza. It should be noted that 'Vaccino antiinfluenzale' gets peak values some weeks before influenza gets the peak, it is expected, since people will do vaccination before the season.  

# 2.2

Now, let's compute correlation and fittness of number of views for each page with Influnet data. 


```python
corr=pd.DataFrame(index=['Influnet'],columns=page_list[:12])
r2=pd.DataFrame(index=['Influnet'],columns=page_list[:12])
for i in page_list:
    if (i!='Influnet'):
        corr[i]['Influnet']=round( numpy.corrcoef(w[i], w["Influnet"])[0, 1],3)
    w[i]=(w[i]-min(w[i]))/(max(w[i])-min(w[i]))    #this normalisation is neccessary to find R square
for i in page_list:    
    if (i!='Influnet'):
        r2[i]['Influnet']=round(sklearn.metrics.r2_score(w[i], w['Influnet']),3)
print("Correlation table:")
display(corr)

print('R square table:')
display(r2)
```

    Correlation table:
    


<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Febbre</th>
      <th>Paracetamolo</th>
      <th>Rinorrea</th>
      <th>Orthomyxoviridae</th>
      <th>Oseltamivir</th>
      <th>Zanamivir</th>
      <th>Ibuprofene</th>
      <th>Cefalea</th>
      <th>Mialgia</th>
      <th>Vomito</th>
      <th>Tosse</th>
      <th>Vaccino antinfluenzale</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Influnet</th>
      <td>0.633</td>
      <td>0.763</td>
      <td>0.296</td>
      <td>0.661</td>
      <td>0.924</td>
      <td>0.816</td>
      <td>0.548</td>
      <td>0.099</td>
      <td>0.147</td>
      <td>0.014</td>
      <td>0.242</td>
      <td>0.013</td>
    </tr>
  </tbody>
</table>
</div>


    R square table:
    


<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Febbre</th>
      <th>Paracetamolo</th>
      <th>Rinorrea</th>
      <th>Orthomyxoviridae</th>
      <th>Oseltamivir</th>
      <th>Zanamivir</th>
      <th>Ibuprofene</th>
      <th>Cefalea</th>
      <th>Mialgia</th>
      <th>Vomito</th>
      <th>Tosse</th>
      <th>Vaccino antinfluenzale</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Influnet</th>
      <td>-0.122</td>
      <td>0.324</td>
      <td>-0.703</td>
      <td>0.118</td>
      <td>0.771</td>
      <td>0.295</td>
      <td>-0.349</td>
      <td>-2.168</td>
      <td>-1.185</td>
      <td>-1.121</td>
      <td>-0.726</td>
      <td>-0.858</td>
    </tr>
  </tbody>
</table>
</div>


The highest correlation is for the page 'Oseltamivir' and this is even higher than page 'Influenza', and also R square for this page is higher. Furthermore, Zanamivir and Paracetamolo also show good correlation, and comparitavely good R square. Among symptom pages, Febbre is comparetavely in good correlation with Influnet. Indeed, people with influenza ,high likely, will look for treatment of the illness, while temperature and other symptoms could indicate to other illnesses. Therefore, we can say pages devoted for treating influenza could be good source of information for predicting influenza incidence.  

# Part 3

# 3.1

For the prediction of the influenza incidence, I will use the model proposed by the article, namely, generalized linear model with poisson distribution using log-link function. Actually, poisson regresson is used for count data, which is the case for the influenza. Furhermore, to prevent overfitting the proposed model will have early stopping, which will control how test and train improves, and reqularization term L1, which will penalize coefficients to some degree alpha. For the loss function, added regularization term will be:

                                        reg_term=alpha*||beta||
  here, ||beta|| - root of squared sum of all coefficients.
This alpha will be hyperparameter of our model, which should be chosen in appropriate way to decrease error. In order to evaluate the performance of the model, root mean squared error (RMSE) metric is used. Due to limitation data, K-Fold cross-validation is conducted, and as number of folds I selected 4,which for 84 weeks gives 21 test set (25%) and 63 train set (75%). For the feature selection, we need to filter out highly mutually correlated features. Therefore, before running model, I will do cross-correlation and drop all features which corr. coefficient higher than 0.8.

I will use XGBoost package for running the model. This package allows us to do extreme gradient boosting algorithm using parallel and fast learning computation. 

To visualize the performance of the model, I will plot real influnet vs predicted results from K-Fold validation by averaging of each fold results.
    


```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import xgboost as xgb
```


```python
copy=v.copy()  # let's copy data from previous part 
```


```python
ff=pd.DataFrame()
ff['Influenza']=comb['Pageview']    
copy=pd.concat([copy, ff], axis=1)         #adding new column Influenza from part 1

```


```python
x=copy[['Febbre', 'Paracetamolo', 'Rinorrea', 'Orthomyxoviridae', 'Oseltamivir', 'Zanamivir', 'Ibuprofene',
          "Cefalea",'Mialgia', 'Vomito','Tosse', 'Vaccino antinfluenzale','Influenza' ]]   #predictors
y=copy[['Influnet']]    #target ouput
copy.head(5)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Week</th>
      <th>Influnet</th>
      <th>Febbre</th>
      <th>Paracetamolo</th>
      <th>Rinorrea</th>
      <th>Orthomyxoviridae</th>
      <th>Oseltamivir</th>
      <th>Zanamivir</th>
      <th>Ibuprofene</th>
      <th>Cefalea</th>
      <th>Mialgia</th>
      <th>Vomito</th>
      <th>Tosse</th>
      <th>Vaccino antinfluenzale</th>
      <th>Influenza</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015</td>
      <td>42</td>
      <td>0.40</td>
      <td>3330</td>
      <td>6318</td>
      <td>500</td>
      <td>181</td>
      <td>109</td>
      <td>43</td>
      <td>3828</td>
      <td>2289</td>
      <td>1106</td>
      <td>1691</td>
      <td>1077</td>
      <td>572</td>
      <td>1400</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015</td>
      <td>43</td>
      <td>0.50</td>
      <td>3307</td>
      <td>6241</td>
      <td>420</td>
      <td>173</td>
      <td>68</td>
      <td>66</td>
      <td>3991</td>
      <td>2256</td>
      <td>980</td>
      <td>1583</td>
      <td>1055</td>
      <td>614</td>
      <td>1317</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015</td>
      <td>44</td>
      <td>0.68</td>
      <td>3150</td>
      <td>5937</td>
      <td>395</td>
      <td>187</td>
      <td>90</td>
      <td>47</td>
      <td>3863</td>
      <td>2274</td>
      <td>944</td>
      <td>1779</td>
      <td>1138</td>
      <td>681</td>
      <td>1226</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015</td>
      <td>45</td>
      <td>0.76</td>
      <td>3305</td>
      <td>6064</td>
      <td>472</td>
      <td>165</td>
      <td>109</td>
      <td>66</td>
      <td>3865</td>
      <td>2516</td>
      <td>1028</td>
      <td>1814</td>
      <td>1140</td>
      <td>842</td>
      <td>1336</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015</td>
      <td>46</td>
      <td>0.88</td>
      <td>3088</td>
      <td>5805</td>
      <td>380</td>
      <td>201</td>
      <td>110</td>
      <td>85</td>
      <td>3911</td>
      <td>2503</td>
      <td>1032</td>
      <td>1884</td>
      <td>1175</td>
      <td>899</td>
      <td>1298</td>
    </tr>
  </tbody>
</table>
</div>




```python
#let's do cross - correlation 
correlation=x.corr(method='pearson').abs()
correlation.style.background_gradient().set_precision(2)
```




<style  type="text/css" >
    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row0_col0 {
            background-color:  #023858;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row0_col1 {
            background-color:  #056dac;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row0_col2 {
            background-color:  #569dc8;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row0_col3 {
            background-color:  #b1c2de;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row0_col4 {
            background-color:  #4e9ac6;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row0_col5 {
            background-color:  #529bc7;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row0_col6 {
            background-color:  #0f76b3;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row0_col7 {
            background-color:  #3790c0;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row0_col8 {
            background-color:  #84b0d3;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row0_col9 {
            background-color:  #589ec8;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row0_col10 {
            background-color:  #0872b1;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row0_col11 {
            background-color:  #f8f1f8;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row0_col12 {
            background-color:  #2987bc;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row1_col0 {
            background-color:  #056faf;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row1_col1 {
            background-color:  #023858;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row1_col2 {
            background-color:  #e0deed;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row1_col3 {
            background-color:  #7dacd1;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row1_col4 {
            background-color:  #0a73b2;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row1_col5 {
            background-color:  #0d75b3;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row1_col6 {
            background-color:  #348ebf;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row1_col7 {
            background-color:  #e4e1ef;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row1_col8 {
            background-color:  #f6eff7;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row1_col9 {
            background-color:  #dcdaeb;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row1_col10 {
            background-color:  #a4bcda;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row1_col11 {
            background-color:  #fff7fb;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row1_col12 {
            background-color:  #045e94;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row2_col0 {
            background-color:  #3991c1;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row2_col1 {
            background-color:  #b9c6e0;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row2_col2 {
            background-color:  #023858;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row2_col3 {
            background-color:  #dedcec;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row2_col4 {
            background-color:  #d3d4e7;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row2_col5 {
            background-color:  #d2d3e7;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row2_col6 {
            background-color:  #9ab8d8;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row2_col7 {
            background-color:  #0c74b2;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row2_col8 {
            background-color:  #056dab;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row2_col9 {
            background-color:  #0570b0;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row2_col10 {
            background-color:  #056ead;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row2_col11 {
            background-color:  #d4d4e8;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row2_col12 {
            background-color:  #d2d3e7;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row3_col0 {
            background-color:  #c2cbe2;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row3_col1 {
            background-color:  #83afd3;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row3_col2 {
            background-color:  #fff7fb;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row3_col3 {
            background-color:  #023858;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row3_col4 {
            background-color:  #187cb6;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row3_col5 {
            background-color:  #1b7eb7;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row3_col6 {
            background-color:  #d0d1e6;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row3_col7 {
            background-color:  #fbf3f9;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row3_col8 {
            background-color:  #fef6fb;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row3_col9 {
            background-color:  #faf3f9;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row3_col10 {
            background-color:  #fff7fb;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row3_col11 {
            background-color:  #f9f2f8;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row3_col12 {
            background-color:  #2f8bbe;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row4_col0 {
            background-color:  #65a3cb;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row4_col1 {
            background-color:  #1077b4;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row4_col2 {
            background-color:  #fbf4f9;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row4_col3 {
            background-color:  #1c7fb8;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row4_col4 {
            background-color:  #023858;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row4_col5 {
            background-color:  #045e94;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row4_col6 {
            background-color:  #adc1dd;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row4_col7 {
            background-color:  #fff7fb;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row4_col8 {
            background-color:  #fdf5fa;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row4_col9 {
            background-color:  #fff7fb;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row4_col10 {
            background-color:  #f2ecf5;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row4_col11 {
            background-color:  #f7f0f7;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row4_col12 {
            background-color:  #04588a;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row5_col0 {
            background-color:  #529bc7;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row5_col1 {
            background-color:  #0771b1;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row5_col2 {
            background-color:  #ece7f2;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row5_col3 {
            background-color:  #1278b4;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row5_col4 {
            background-color:  #045b8f;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row5_col5 {
            background-color:  #023858;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row5_col6 {
            background-color:  #86b0d3;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row5_col7 {
            background-color:  #efe9f3;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row5_col8 {
            background-color:  #f8f1f8;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row5_col9 {
            background-color:  #f0eaf4;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row5_col10 {
            background-color:  #d8d7e9;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row5_col11 {
            background-color:  #dcdaeb;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row5_col12 {
            background-color:  #04598c;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row6_col0 {
            background-color:  #056faf;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row6_col1 {
            background-color:  #1e80b8;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row6_col2 {
            background-color:  #a1bbda;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row6_col3 {
            background-color:  #a5bddb;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row6_col4 {
            background-color:  #7dacd1;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row6_col5 {
            background-color:  #6da6cd;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row6_col6 {
            background-color:  #023858;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row6_col7 {
            background-color:  #76aad0;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row6_col8 {
            background-color:  #ced0e6;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row6_col9 {
            background-color:  #abbfdc;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row6_col10 {
            background-color:  #308cbe;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row6_col11 {
            background-color:  #e7e3f0;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row6_col12 {
            background-color:  #3d93c2;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row7_col0 {
            background-color:  #3d93c2;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row7_col1 {
            background-color:  #e0dded;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row7_col2 {
            background-color:  #2081b9;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row7_col3 {
            background-color:  #f4eef6;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row7_col4 {
            background-color:  #f4edf6;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row7_col5 {
            background-color:  #f2ecf5;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row7_col6 {
            background-color:  #94b6d7;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row7_col7 {
            background-color:  #023858;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row7_col8 {
            background-color:  #056ba9;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row7_col9 {
            background-color:  #157ab5;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row7_col10 {
            background-color:  #0570b0;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row7_col11 {
            background-color:  #f8f1f8;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row7_col12 {
            background-color:  #f1ebf5;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row8_col0 {
            background-color:  #8eb3d5;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row8_col1 {
            background-color:  #f5eff6;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row8_col2 {
            background-color:  #1077b4;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row8_col3 {
            background-color:  #faf3f9;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row8_col4 {
            background-color:  #f4edf6;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row8_col5 {
            background-color:  #fff7fb;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row8_col6 {
            background-color:  #ebe6f2;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row8_col7 {
            background-color:  #056caa;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row8_col8 {
            background-color:  #023858;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row8_col9 {
            background-color:  #056faf;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row8_col10 {
            background-color:  #3b92c1;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row8_col11 {
            background-color:  #f4edf6;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row8_col12 {
            background-color:  #f9f2f8;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row9_col0 {
            background-color:  #6fa7ce;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row9_col1 {
            background-color:  #e7e3f0;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row9_col2 {
            background-color:  #2182b9;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row9_col3 {
            background-color:  #fff7fb;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row9_col4 {
            background-color:  #fff7fb;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row9_col5 {
            background-color:  #fff7fb;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row9_col6 {
            background-color:  #d9d8ea;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row9_col7 {
            background-color:  #1e80b8;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row9_col8 {
            background-color:  #0a73b2;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row9_col9 {
            background-color:  #023858;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row9_col10 {
            background-color:  #045c90;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row9_col11 {
            background-color:  #fff7fb;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row9_col12 {
            background-color:  #fff7fb;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row10_col0 {
            background-color:  #0771b1;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row10_col1 {
            background-color:  #97b7d7;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row10_col2 {
            background-color:  #0f76b3;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row10_col3 {
            background-color:  #f4eef6;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row10_col4 {
            background-color:  #dfddec;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row10_col5 {
            background-color:  #d7d6e9;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row10_col6 {
            background-color:  #4496c3;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row10_col7 {
            background-color:  #056fae;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row10_col8 {
            background-color:  #328dbf;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row10_col9 {
            background-color:  #04598c;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row10_col10 {
            background-color:  #023858;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row10_col11 {
            background-color:  #f1ebf5;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row10_col12 {
            background-color:  #c8cde4;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row11_col0 {
            background-color:  #fff7fb;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row11_col1 {
            background-color:  #fff7fb;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row11_col2 {
            background-color:  #f4eef6;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row11_col3 {
            background-color:  #f5eff6;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row11_col4 {
            background-color:  #eee9f3;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row11_col5 {
            background-color:  #e5e1ef;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row11_col6 {
            background-color:  #fff7fb;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row11_col7 {
            background-color:  #fbf4f9;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row11_col8 {
            background-color:  #f4eef6;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row11_col9 {
            background-color:  #f7f0f7;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row11_col10 {
            background-color:  #f8f1f8;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row11_col11 {
            background-color:  #023858;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row11_col12 {
            background-color:  #e4e1ef;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row12_col0 {
            background-color:  #358fc0;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row12_col1 {
            background-color:  #046096;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row12_col2 {
            background-color:  #f8f1f8;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row12_col3 {
            background-color:  #308cbe;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row12_col4 {
            background-color:  #045788;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row12_col5 {
            background-color:  #045b8f;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row12_col6 {
            background-color:  #67a4cc;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row12_col7 {
            background-color:  #faf2f8;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row12_col8 {
            background-color:  #fff7fb;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row12_col9 {
            background-color:  #fdf5fa;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row12_col10 {
            background-color:  #dad9ea;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row12_col11 {
            background-color:  #ebe6f2;
        }    #T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row12_col12 {
            background-color:  #023858;
        }</style>  
<table id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8" > 
<thead>    <tr> 
        <th class="blank level0" ></th> 
        <th class="col_heading level0 col0" >Febbre</th> 
        <th class="col_heading level0 col1" >Paracetamolo</th> 
        <th class="col_heading level0 col2" >Rinorrea</th> 
        <th class="col_heading level0 col3" >Orthomyxoviridae</th> 
        <th class="col_heading level0 col4" >Oseltamivir</th> 
        <th class="col_heading level0 col5" >Zanamivir</th> 
        <th class="col_heading level0 col6" >Ibuprofene</th> 
        <th class="col_heading level0 col7" >Cefalea</th> 
        <th class="col_heading level0 col8" >Mialgia</th> 
        <th class="col_heading level0 col9" >Vomito</th> 
        <th class="col_heading level0 col10" >Tosse</th> 
        <th class="col_heading level0 col11" >Vaccino antinfluenzale</th> 
        <th class="col_heading level0 col12" >Influenza</th> 
    </tr></thead> 
<tbody>    <tr> 
        <th id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8level0_row0" class="row_heading level0 row0" >Febbre</th> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row0_col0" class="data row0 col0" >1</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row0_col1" class="data row0 col1" >0.78</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row0_col2" class="data row0 col2" >0.66</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row0_col3" class="data row0 col3" >0.37</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row0_col4" class="data row0 col4" >0.58</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row0_col5" class="data row0 col5" >0.62</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row0_col6" class="data row0 col6" >0.78</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row0_col7" class="data row0 col7" >0.65</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row0_col8" class="data row0 col8" >0.5</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row0_col9" class="data row0 col9" >0.56</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row0_col10" class="data row0 col10" >0.77</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row0_col11" class="data row0 col11" >0.11</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row0_col12" class="data row0 col12" >0.67</td> 
    </tr>    <tr> 
        <th id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8level0_row1" class="row_heading level0 row1" >Paracetamolo</th> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row1_col0" class="data row1 col0" >0.78</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row1_col1" class="data row1 col1" >1</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row1_col2" class="data row1 col2" >0.36</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row1_col3" class="data row1 col3" >0.5</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row1_col4" class="data row1 col4" >0.74</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row1_col5" class="data row1 col5" >0.76</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row1_col6" class="data row1 col6" >0.71</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row1_col7" class="data row1 col7" >0.23</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row1_col8" class="data row1 col8" >0.12</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row1_col9" class="data row1 col9" >0.21</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row1_col10" class="data row1 col10" >0.45</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row1_col11" class="data row1 col11" >0.064</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row1_col12" class="data row1 col12" >0.85</td> 
    </tr>    <tr> 
        <th id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8level0_row2" class="row_heading level0 row2" >Rinorrea</th> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row2_col0" class="data row2 col0" >0.66</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row2_col1" class="data row2 col1" >0.36</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row2_col2" class="data row2 col2" >1</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row2_col3" class="data row2 col3" >0.22</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row2_col4" class="data row2 col4" >0.24</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row2_col5" class="data row2 col5" >0.32</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row2_col6" class="data row2 col6" >0.53</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row2_col7" class="data row2 col7" >0.75</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row2_col8" class="data row2 col8" >0.78</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row2_col9" class="data row2 col9" >0.75</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row2_col10" class="data row2 col10" >0.79</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row2_col11" class="data row2 col11" >0.28</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row2_col12" class="data row2 col12" >0.26</td> 
    </tr>    <tr> 
        <th id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8level0_row3" class="row_heading level0 row3" >Orthomyxoviridae</th> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row3_col0" class="data row3 col0" >0.37</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row3_col1" class="data row3 col1" >0.5</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row3_col2" class="data row3 col2" >0.22</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row3_col3" class="data row3 col3" >1</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row3_col4" class="data row3 col4" >0.7</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row3_col5" class="data row3 col5" >0.73</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row3_col6" class="data row3 col6" >0.4</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row3_col7" class="data row3 col7" >0.11</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row3_col8" class="data row3 col8" >0.074</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row3_col9" class="data row3 col9" >0.041</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row3_col10" class="data row3 col10" >0.11</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row3_col11" class="data row3 col11" >0.1</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row3_col12" class="data row3 col12" >0.65</td> 
    </tr>    <tr> 
        <th id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8level0_row4" class="row_heading level0 row4" >Oseltamivir</th> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row4_col0" class="data row4 col0" >0.58</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row4_col1" class="data row4 col1" >0.74</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row4_col2" class="data row4 col2" >0.24</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row4_col3" class="data row4 col3" >0.7</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row4_col4" class="data row4 col4" >1</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row4_col5" class="data row4 col5" >0.87</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row4_col6" class="data row4 col6" >0.48</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row4_col7" class="data row4 col7" >0.084</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row4_col8" class="data row4 col8" >0.085</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row4_col9" class="data row4 col9" >0.0091</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row4_col10" class="data row4 col10" >0.19</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row4_col11" class="data row4 col11" >0.12</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row4_col12" class="data row4 col12" >0.88</td> 
    </tr>    <tr> 
        <th id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8level0_row5" class="row_heading level0 row5" >Zanamivir</th> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row5_col0" class="data row5 col0" >0.62</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row5_col1" class="data row5 col1" >0.76</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row5_col2" class="data row5 col2" >0.32</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row5_col3" class="data row5 col3" >0.73</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row5_col4" class="data row5 col4" >0.87</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row5_col5" class="data row5 col5" >1</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row5_col6" class="data row5 col6" >0.57</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row5_col7" class="data row5 col7" >0.18</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row5_col8" class="data row5 col8" >0.11</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row5_col9" class="data row5 col9" >0.11</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row5_col10" class="data row5 col10" >0.31</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row5_col11" class="data row5 col11" >0.25</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row5_col12" class="data row5 col12" >0.88</td> 
    </tr>    <tr> 
        <th id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8level0_row6" class="row_heading level0 row6" >Ibuprofene</th> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row6_col0" class="data row6 col0" >0.78</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row6_col1" class="data row6 col1" >0.71</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row6_col2" class="data row6 col2" >0.53</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row6_col3" class="data row6 col3" >0.4</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row6_col4" class="data row6 col4" >0.48</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row6_col5" class="data row6 col5" >0.57</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row6_col6" class="data row6 col6" >1</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row6_col7" class="data row6 col7" >0.54</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row6_col8" class="data row6 col8" >0.31</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row6_col9" class="data row6 col9" >0.37</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row6_col10" class="data row6 col10" >0.68</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row6_col11" class="data row6 col11" >0.2</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row6_col12" class="data row6 col12" >0.62</td> 
    </tr>    <tr> 
        <th id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8level0_row7" class="row_heading level0 row7" >Cefalea</th> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row7_col0" class="data row7 col0" >0.65</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row7_col1" class="data row7 col1" >0.23</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row7_col2" class="data row7 col2" >0.75</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row7_col3" class="data row7 col3" >0.11</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row7_col4" class="data row7 col4" >0.084</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row7_col5" class="data row7 col5" >0.18</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row7_col6" class="data row7 col6" >0.54</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row7_col7" class="data row7 col7" >1</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row7_col8" class="data row7 col8" >0.79</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row7_col9" class="data row7 col9" >0.71</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row7_col10" class="data row7 col10" >0.77</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row7_col11" class="data row7 col11" >0.11</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row7_col12" class="data row7 col12" >0.12</td> 
    </tr>    <tr> 
        <th id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8level0_row8" class="row_heading level0 row8" >Mialgia</th> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row8_col0" class="data row8 col0" >0.5</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row8_col1" class="data row8 col1" >0.12</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row8_col2" class="data row8 col2" >0.78</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row8_col3" class="data row8 col3" >0.074</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row8_col4" class="data row8 col4" >0.085</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row8_col5" class="data row8 col5" >0.11</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row8_col6" class="data row8 col6" >0.31</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row8_col7" class="data row8 col7" >0.79</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row8_col8" class="data row8 col8" >1</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row8_col9" class="data row8 col9" >0.75</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row8_col10" class="data row8 col10" >0.66</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row8_col11" class="data row8 col11" >0.13</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row8_col12" class="data row8 col12" >0.067</td> 
    </tr>    <tr> 
        <th id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8level0_row9" class="row_heading level0 row9" >Vomito</th> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row9_col0" class="data row9 col0" >0.56</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row9_col1" class="data row9 col1" >0.21</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row9_col2" class="data row9 col2" >0.75</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row9_col3" class="data row9 col3" >0.041</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row9_col4" class="data row9 col4" >0.0091</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row9_col5" class="data row9 col5" >0.11</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row9_col6" class="data row9 col6" >0.37</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row9_col7" class="data row9 col7" >0.71</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row9_col8" class="data row9 col8" >0.75</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row9_col9" class="data row9 col9" >1</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row9_col10" class="data row9 col10" >0.88</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row9_col11" class="data row9 col11" >0.067</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row9_col12" class="data row9 col12" >0.026</td> 
    </tr>    <tr> 
        <th id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8level0_row10" class="row_heading level0 row10" >Tosse</th> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row10_col0" class="data row10 col0" >0.77</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row10_col1" class="data row10 col1" >0.45</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row10_col2" class="data row10 col2" >0.79</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row10_col3" class="data row10 col3" >0.11</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row10_col4" class="data row10 col4" >0.19</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row10_col5" class="data row10 col5" >0.31</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row10_col6" class="data row10 col6" >0.68</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row10_col7" class="data row10 col7" >0.77</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row10_col8" class="data row10 col8" >0.66</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row10_col9" class="data row10 col9" >0.88</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row10_col10" class="data row10 col10" >1</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row10_col11" class="data row10 col11" >0.15</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row10_col12" class="data row10 col12" >0.3</td> 
    </tr>    <tr> 
        <th id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8level0_row11" class="row_heading level0 row11" >Vaccino antinfluenzale</th> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row11_col0" class="data row11 col0" >0.11</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row11_col1" class="data row11 col1" >0.064</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row11_col2" class="data row11 col2" >0.28</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row11_col3" class="data row11 col3" >0.1</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row11_col4" class="data row11 col4" >0.12</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row11_col5" class="data row11 col5" >0.25</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row11_col6" class="data row11 col6" >0.2</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row11_col7" class="data row11 col7" >0.11</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row11_col8" class="data row11 col8" >0.13</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row11_col9" class="data row11 col9" >0.067</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row11_col10" class="data row11 col10" >0.15</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row11_col11" class="data row11 col11" >1</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row11_col12" class="data row11 col12" >0.19</td> 
    </tr>    <tr> 
        <th id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8level0_row12" class="row_heading level0 row12" >Influenza</th> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row12_col0" class="data row12 col0" >0.67</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row12_col1" class="data row12 col1" >0.85</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row12_col2" class="data row12 col2" >0.26</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row12_col3" class="data row12 col3" >0.65</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row12_col4" class="data row12 col4" >0.88</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row12_col5" class="data row12 col5" >0.88</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row12_col6" class="data row12 col6" >0.62</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row12_col7" class="data row12 col7" >0.12</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row12_col8" class="data row12 col8" >0.067</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row12_col9" class="data row12 col9" >0.026</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row12_col10" class="data row12 col10" >0.3</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row12_col11" class="data row12 col11" >0.19</td> 
        <td id="T_dd1c4e4c_fd7e_11e8_bc78_1cb72c95aad8row12_col12" class="data row12 col12" >1</td> 
    </tr></tbody> 
</table> 



There are high correlations for Paracetamolo, Influenza, Tosse and etc.
Let's remove them


```python
# choose upper triangle of correlation matrix
upp= correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(np.bool))

# Find and drop features which have correlation coeff higher than 0.8
to_drop = [column for column in upp.columns if any(upp[column] > 0.8)]
x=x.drop(x[to_drop], axis=1)
x.head(5)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Febbre</th>
      <th>Paracetamolo</th>
      <th>Rinorrea</th>
      <th>Orthomyxoviridae</th>
      <th>Oseltamivir</th>
      <th>Ibuprofene</th>
      <th>Cefalea</th>
      <th>Mialgia</th>
      <th>Vomito</th>
      <th>Vaccino antinfluenzale</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3330</td>
      <td>6318</td>
      <td>500</td>
      <td>181</td>
      <td>109</td>
      <td>3828</td>
      <td>2289</td>
      <td>1106</td>
      <td>1691</td>
      <td>572</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3307</td>
      <td>6241</td>
      <td>420</td>
      <td>173</td>
      <td>68</td>
      <td>3991</td>
      <td>2256</td>
      <td>980</td>
      <td>1583</td>
      <td>614</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3150</td>
      <td>5937</td>
      <td>395</td>
      <td>187</td>
      <td>90</td>
      <td>3863</td>
      <td>2274</td>
      <td>944</td>
      <td>1779</td>
      <td>681</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3305</td>
      <td>6064</td>
      <td>472</td>
      <td>165</td>
      <td>109</td>
      <td>3865</td>
      <td>2516</td>
      <td>1028</td>
      <td>1814</td>
      <td>842</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3088</td>
      <td>5805</td>
      <td>380</td>
      <td>201</td>
      <td>110</td>
      <td>3911</td>
      <td>2503</td>
      <td>1032</td>
      <td>1884</td>
      <td>899</td>
    </tr>
  </tbody>
</table>
</div>




```python
#number of folds = 4, random state as last 4 digits of my ID
K = 4
kf = KFold(n_splits = K, random_state = 1856, shuffle = True)  
```


```python
train = np.array(x)    #our variables
target_train = np.array(y)   #target output
```


```python
#our model
def poiss_model(train, target_train, kf, alpha):   
    
    xgb_preds=[]      #to store predicted values
    progress=dict()   #to see at each iteration RMSE values
    error=[]
    for train_index, test_index in kf.split(train):

        

        #Cross-validation
        train_X, valid_X = train[train_index], train[test_index]       
        train_y, valid_y = target_train[train_index], target_train[test_index]

        #parameters of model: Poisson regression, evaluation metric RMSE and regulation alpha 
        params = {'objective': 'count:poisson', 'seed': 1856, 'silent': 1, 'eval_metric':'rmse', 'reg_alpha':alpha}

        #convert to appropriate form for XGBoost
        d_train = xgb.DMatrix(train_X, train_y)
        d_valid = xgb.DMatrix(valid_X, valid_y)
        d_test = xgb.DMatrix(train)

        watchlist = [(d_train, 'train'), (d_valid, 'valid')]   #monitor these values

        model = xgb.train(params, d_train, 200,  watchlist, maximize=True, verbose_eval=False, early_stopping_rounds=50,
                         evals_result=progress)      #iteration number = 200

        error.append(progress['valid']['rmse'][-1])  #the final RMSE value of the model
        
        xgb_pred = model.predict(d_test)      #prediction 
        xgb_preds.append(list(xgb_pred))      #store prediction
    return (model, xgb_preds, error)            #return the model, predicted values and error
```


```python
model, pred, error=poiss_model(train, target_train, kf, 0.1)   #running model with alpha = 0.1
```

In considering RMSE value, we need to evaluate how the RMSE is good, since it is not relative value and it depends on range of target output of what we are considering. In this case, good option could be computing normalized RMSE value relative to target output range, and it will give us intuitive evaluation of model. So:
                                
                                norm_RMSE= RMSE/(Influnet_max-Influnet_min)
                            
  
                               


```python
print('RMSE :',round(np.mean(np.array(error)),2))
print('Nomalized RMSE as %:',round(100*np.mean(np.array(error))/(max(copy['Influnet'])-min(copy['Influnet'])),2))
```

    RMSE : 1.36
    Nomalized RMSE as %: 9.5
    

So, for alpha = 0.1, the model shows norm RMSE - 9.5%. Since alpha is tuning parameter, we need to search for the best value of alpha. 


```python
#range of possible alpha values from 0 to 1 by 0.02 step
rg=numpy.arange(0, 1, 0.02)

results=[]   #
for alpha in rg:
    model, pred,error=poiss_model(train, target_train, kf, alpha)
    results.append(round(100*np.mean(np.array(error))/(max(copy['Influnet'])-min(copy['Influnet'])),2))
```


```python
plt.plot(rg,results, color='c',marker='o', linestyle='dashed', linewidth=0.8)
plt.title('Norm_RMSE of model vs alpha')
plt.xlabel('alpha')
plt.ylabel('norm_RMSE')
plt.show()
```


![png](output_64_0.png)


So, from the plot, the best value for alpha is 0.04


```python
#re running the model for alpha = 0.04
model, pred, error=poiss_model(train, target_train, kf, 0.04)
print('RMSE :',round(np.mean(np.array(error)),2))
print('Nomalized RMSE as %:',round(100*np.mean(np.array(error))/(max(copy['Influnet'])-min(copy['Influnet'])),2))
```

    RMSE : 1.32
    Nomalized RMSE as %: 9.18
    

Now, let's plot Influnet vs predicted from cross validation 


```python

pp=np.array(pred)   #predicted results from the model
predicted=np.mean(pp, axis=0)     #taking mean value for each fold predicted values
copy['Predicted']=predicted           #add new column for predicted
index=[]
index=copy['Year'].apply(str)+"-"+copy['Week'].apply(str)

ax=copy.plot(x=index, y=['Influnet','Predicted'], title=" Predicted vs Influnet",rot=90, figsize=(10,5), legend=True,
            kind="line", xticks=range(0, 84, 5))


ax.set_xlabel("Year-Week")
ax.set_ylabel("case per 1000")
ax.grid( linewidth='0.2', color='grey')
plt.show()
```


![png](output_68_0.png)


So, our model showed RMSE = 1.32 and norm_RMSE=9.18% with alpha being equal to 0.04. Plot above also indicates to good fitness with real data. For the first and second peak of influenza, the model behaves good, but at the last peak point it underestimates the incidence.

# 3.2

In this part, we have to predict the incidence for the target week having data for Influnet and all Wiki pages for the preceding week. It means I have to shift rows for influnet and all pages, while keeping Influnet as new target output


```python
target=pd.DataFrame()
preceding=pd.DataFrame()
```


```python
#the target week
target[['Year', 'Week', 'Influnet']]=copy[['Year', 'Week', 'Influnet']].iloc[1:].reset_index(drop=True)   

#preceding week 
preceding[['In_pred','Febbre', 'Paracetamolo', 'Rinorrea', 'Orthomyxoviridae', 'Oseltamivir', 'Zanamivir', 'Ibuprofene',
          "Cefalea",'Mialgia', 'Vomito','Tosse', 'Vaccino antinfluenzale', 'Influenza' ]]=copy[['Influnet','Febbre', 'Paracetamolo', 'Rinorrea', 'Orthomyxoviridae', 'Oseltamivir', 'Zanamivir', 'Ibuprofene',
          "Cefalea",'Mialgia', 'Vomito','Tosse', 'Vaccino antinfluenzale', 'Influenza' ]].iloc[0:83].reset_index(drop=True)
```


```python
new_data=pd.concat([target, preceding], axis=1)
new_data.head(5)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Week</th>
      <th>Influnet</th>
      <th>In_pred</th>
      <th>Febbre</th>
      <th>Paracetamolo</th>
      <th>Rinorrea</th>
      <th>Orthomyxoviridae</th>
      <th>Oseltamivir</th>
      <th>Zanamivir</th>
      <th>Ibuprofene</th>
      <th>Cefalea</th>
      <th>Mialgia</th>
      <th>Vomito</th>
      <th>Tosse</th>
      <th>Vaccino antinfluenzale</th>
      <th>Influenza</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015</td>
      <td>43</td>
      <td>0.50</td>
      <td>0.40</td>
      <td>3330</td>
      <td>6318</td>
      <td>500</td>
      <td>181</td>
      <td>109</td>
      <td>43</td>
      <td>3828</td>
      <td>2289</td>
      <td>1106</td>
      <td>1691</td>
      <td>1077</td>
      <td>572</td>
      <td>1400</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015</td>
      <td>44</td>
      <td>0.68</td>
      <td>0.50</td>
      <td>3307</td>
      <td>6241</td>
      <td>420</td>
      <td>173</td>
      <td>68</td>
      <td>66</td>
      <td>3991</td>
      <td>2256</td>
      <td>980</td>
      <td>1583</td>
      <td>1055</td>
      <td>614</td>
      <td>1317</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015</td>
      <td>45</td>
      <td>0.76</td>
      <td>0.68</td>
      <td>3150</td>
      <td>5937</td>
      <td>395</td>
      <td>187</td>
      <td>90</td>
      <td>47</td>
      <td>3863</td>
      <td>2274</td>
      <td>944</td>
      <td>1779</td>
      <td>1138</td>
      <td>681</td>
      <td>1226</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015</td>
      <td>46</td>
      <td>0.88</td>
      <td>0.76</td>
      <td>3305</td>
      <td>6064</td>
      <td>472</td>
      <td>165</td>
      <td>109</td>
      <td>66</td>
      <td>3865</td>
      <td>2516</td>
      <td>1028</td>
      <td>1814</td>
      <td>1140</td>
      <td>842</td>
      <td>1336</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015</td>
      <td>47</td>
      <td>1.02</td>
      <td>0.88</td>
      <td>3088</td>
      <td>5805</td>
      <td>380</td>
      <td>201</td>
      <td>110</td>
      <td>85</td>
      <td>3911</td>
      <td>2503</td>
      <td>1032</td>
      <td>1884</td>
      <td>1175</td>
      <td>899</td>
      <td>1298</td>
    </tr>
  </tbody>
</table>
</div>




```python
#creating training set
x=new_data[['In_pred','Febbre', 'Paracetamolo', 'Rinorrea', 'Orthomyxoviridae', 'Oseltamivir', 'Zanamivir', 'Ibuprofene',
          "Cefalea",'Mialgia', 'Vomito','Tosse', 'Vaccino antinfluenzale', 'Influenza' ]]

#creating target output
y=new_data[['Influnet']]
```


```python
#correlation of features
correlation=x.corr(method='pearson').abs()

upp= correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(np.bool))

# Find and drop features which have correlation coeff higher than 0.8
to_drop = [column for column in upp.columns if any(upp[column] > 0.8)]
x=x.drop(x[to_drop], axis=1)
```


```python
train = np.array(x)
target_train = np.array(y)
```

Since we have new data for training our mode, we have to choose appropriate value for alpha in the same as in previous part


```python
#range of possible alpha values from 0 to 1 by 0.02 step
rg=numpy.arange(0, 1, 0.02)

results=[]   #
for alpha in rg:
    model, pred,error=poiss_model(train, target_train, kf, alpha)
    results.append(round(100*np.mean(np.array(error))/(max(new_data['Influnet'])-min(new_data['Influnet'])),2))
```


```python
plt.plot(rg,results, color='c',marker='o', linestyle='dashed', linewidth=0.8)
plt.title('Norm_RMSE of model vs alpha')
plt.xlabel('alpha')
plt.ylabel('norm_RMSE')
plt.show()
```


![png](output_80_0.png)


So, the best value for the alpha is 0.16


```python
model_new, pred, error=poiss_model(train, target_train, kf, 0.16)
print('RMSE :',round(np.mean(np.array(error)),2))
print('Nomalized RMSE as %:',round(100*np.mean(np.array(error))/(max(new_data['Influnet'])-min(new_data['Influnet'])),2))
```

    RMSE : 1.29
    Nomalized RMSE as %: 9.04
    


```python
pp=np.array(pred)   #predicted results from the model
predicted=np.mean(pp, axis=0)     #taking mean value for each fold predicted values
new_data['Predicted']=predicted           #add new column for predicted
index=[]
index=new_data['Year'].apply(str)+"-"+new_data['Week'].apply(str)

ax=new_data.plot(x=index, y=['Influnet','Predicted'], title=" New model Predicted vs Influnet",rot=90, figsize=(10,5), legend=True,
            kind="line", xticks=range(0, 83, 5))


ax.set_xlabel("Year-Week")
ax.set_ylabel("case per 1000")
ax.grid( linewidth='0.2', color='grey')
plt.show()
```


![png](output_83_0.png)


This new model shows slightly more accurate result than previous model. Final RMSE value is 1.29 and norm_RMSE is 9.04%, getting accurate by 2%. Finally, this model can be improved further by adding new pages from Wikipedia/Google Trend and other sources, and increasing number of data, since I have data for only 84 weeks. 
