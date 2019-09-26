```python
# Dependencies and Setup
%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from scipy.stats import sem

# Hide warning messages in notebook
import warnings
warnings.filterwarnings('ignore')

# File to Load (Remember to Change These)
mouse_data_file = "data/mouse_drug_data.csv"
clinical_trial_file = "data/clinicaltrial_data.csv"

# Read the Mouse and Drug Data and the Clinical Trial Data
mouse_data_df = pd.read_csv(mouse_data_file)
clinical_trial_df = pd.read_csv(clinical_trial_file)

# Combine the data into a single dataset

combined_trial_df = pd.merge(clinical_trial_df,mouse_data_df, 
                                 how='outer', on='Mouse ID')
# Display the data table for preview
combined_trial_df.head()

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
      <th>Mouse ID</th>
      <th>Timepoint</th>
      <th>Tumor Volume (mm3)</th>
      <th>Metastatic Sites</th>
      <th>Drug</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>b128</td>
      <td>0</td>
      <td>45.000000</td>
      <td>0</td>
      <td>Capomulin</td>
    </tr>
    <tr>
      <td>1</td>
      <td>b128</td>
      <td>5</td>
      <td>45.651331</td>
      <td>0</td>
      <td>Capomulin</td>
    </tr>
    <tr>
      <td>2</td>
      <td>b128</td>
      <td>10</td>
      <td>43.270852</td>
      <td>0</td>
      <td>Capomulin</td>
    </tr>
    <tr>
      <td>3</td>
      <td>b128</td>
      <td>15</td>
      <td>43.784893</td>
      <td>0</td>
      <td>Capomulin</td>
    </tr>
    <tr>
      <td>4</td>
      <td>b128</td>
      <td>20</td>
      <td>42.731552</td>
      <td>0</td>
      <td>Capomulin</td>
    </tr>
  </tbody>
</table>
</div>



## Tumor Response to Treatment


```python
# Store the Mean Tumor Volume Data Grouped by Drug and Timepoint 
trial_data = combined_trial_df[['Drug', 'Timepoint', 'Tumor Volume (mm3)']]

# Convert to DataFrame

total_volume_data =trial_data.groupby(['Drug','Timepoint'], as_index = False)['Tumor Volume (mm3)'].mean()
tumor_response_data = pd.DataFrame(total_volume_data)

# Preview DataFrame

tumor_response_data.head(5)



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
      <th>Drug</th>
      <th>Timepoint</th>
      <th>Tumor Volume (mm3)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Capomulin</td>
      <td>0</td>
      <td>45.000000</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Capomulin</td>
      <td>5</td>
      <td>44.266086</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Capomulin</td>
      <td>10</td>
      <td>43.084291</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Capomulin</td>
      <td>15</td>
      <td>42.064317</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Capomulin</td>
      <td>20</td>
      <td>40.716325</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Store the Standard Error of Tumor Volumes Grouped by Drug and Timepoint
error_data = combined_trial_df[['Drug', 'Timepoint', 'Tumor Volume (mm3)']]
mean_tumor_volume_ste = combined_trial_df.groupby(['Drug','Timepoint'])['Tumor Volume (mm3)'].sem()
error_data_df = pd.DataFrame(mean_tumor_volume_ste).reset_index()
# Convert to DataFrame

# result = error_data.groupby(['Drug','Timepoint'], as_index=False).agg(
#                       {'Tumor Volume (mm3)':'sem'})
# error_data_df = pd.DataFrame(result)
error_data_df.head()
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
      <th>Drug</th>
      <th>Timepoint</th>
      <th>Tumor Volume (mm3)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Capomulin</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Capomulin</td>
      <td>5</td>
      <td>0.448593</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Capomulin</td>
      <td>10</td>
      <td>0.702684</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Capomulin</td>
      <td>15</td>
      <td>0.838617</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Capomulin</td>
      <td>20</td>
      <td>0.909731</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Minor Data Munging to Re-Format the Data Frames

# df.pivot_table('no of medals', ['Year', 'Country'], 'medal')
mean_data_pivot_table = combined_trial_df.pivot_table('Tumor Volume (mm3)', ['Timepoint'], 'Drug')

# Preview that Reformatting worked
mean_data_pivot_table.head()

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
      <th>Drug</th>
      <th>Capomulin</th>
      <th>Ceftamin</th>
      <th>Infubinol</th>
      <th>Ketapril</th>
      <th>Naftisol</th>
      <th>Placebo</th>
      <th>Propriva</th>
      <th>Ramicane</th>
      <th>Stelasyn</th>
      <th>Zoniferol</th>
    </tr>
    <tr>
      <th>Timepoint</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>45.000000</td>
      <td>45.000000</td>
      <td>45.000000</td>
      <td>45.000000</td>
      <td>45.000000</td>
      <td>45.000000</td>
      <td>45.000000</td>
      <td>45.000000</td>
      <td>45.000000</td>
      <td>45.000000</td>
    </tr>
    <tr>
      <td>5</td>
      <td>44.266086</td>
      <td>46.503051</td>
      <td>47.062001</td>
      <td>47.389175</td>
      <td>46.796098</td>
      <td>47.125589</td>
      <td>47.248967</td>
      <td>43.944859</td>
      <td>47.527452</td>
      <td>46.851818</td>
    </tr>
    <tr>
      <td>10</td>
      <td>43.084291</td>
      <td>48.285125</td>
      <td>49.403909</td>
      <td>49.582269</td>
      <td>48.694210</td>
      <td>49.423329</td>
      <td>49.101541</td>
      <td>42.531957</td>
      <td>49.463844</td>
      <td>48.689881</td>
    </tr>
    <tr>
      <td>15</td>
      <td>42.064317</td>
      <td>50.094055</td>
      <td>51.296397</td>
      <td>52.399974</td>
      <td>50.933018</td>
      <td>51.359742</td>
      <td>51.067318</td>
      <td>41.495061</td>
      <td>51.529409</td>
      <td>50.779059</td>
    </tr>
    <tr>
      <td>20</td>
      <td>40.716325</td>
      <td>52.157049</td>
      <td>53.197691</td>
      <td>54.920935</td>
      <td>53.644087</td>
      <td>54.364417</td>
      <td>53.346737</td>
      <td>40.238325</td>
      <td>54.067395</td>
      <td>53.170334</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Minor Data Munging to Re-Format the Data Frames

# df.pivot_table('no of medals', ['Year', 'Country'], 'medal')
sem_error_pivot_table = error_data_df.pivot_table('Tumor Volume (mm3)', ['Timepoint'], 'Drug')

# Preview that Reformatting worked
sem_error_pivot_table.head()
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
      <th>Drug</th>
      <th>Capomulin</th>
      <th>Ceftamin</th>
      <th>Infubinol</th>
      <th>Ketapril</th>
      <th>Naftisol</th>
      <th>Placebo</th>
      <th>Propriva</th>
      <th>Ramicane</th>
      <th>Stelasyn</th>
      <th>Zoniferol</th>
    </tr>
    <tr>
      <th>Timepoint</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.448593</td>
      <td>0.164505</td>
      <td>0.235102</td>
      <td>0.264819</td>
      <td>0.202385</td>
      <td>0.218091</td>
      <td>0.231708</td>
      <td>0.482955</td>
      <td>0.239862</td>
      <td>0.188950</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.702684</td>
      <td>0.236144</td>
      <td>0.282346</td>
      <td>0.357421</td>
      <td>0.319415</td>
      <td>0.402064</td>
      <td>0.376195</td>
      <td>0.720225</td>
      <td>0.433678</td>
      <td>0.263949</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.838617</td>
      <td>0.332053</td>
      <td>0.357705</td>
      <td>0.580268</td>
      <td>0.444378</td>
      <td>0.614461</td>
      <td>0.466109</td>
      <td>0.770432</td>
      <td>0.493261</td>
      <td>0.370544</td>
    </tr>
    <tr>
      <td>20</td>
      <td>0.909731</td>
      <td>0.359482</td>
      <td>0.476210</td>
      <td>0.726484</td>
      <td>0.595260</td>
      <td>0.839609</td>
      <td>0.555181</td>
      <td>0.786199</td>
      <td>0.621889</td>
      <td>0.533182</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Generate the Plot (with Error Bars)
sem_pivot = sem_error_pivot_table[['Capomulin', 'Infubinol','Ketapril','Placebo']]
drug_data = mean_data_pivot_table[['Capomulin', 'Infubinol','Ketapril','Placebo']]
ax1 = drug_data['Capomulin'].plot(kind='line', yerr = sem_pivot['Capomulin'], linewidth=1, marker= 'o', color='red', label ='Capomulin')
ax2 = drug_data['Infubinol'].plot(kind='line',yerr = sem_pivot['Infubinol'],linewidth=1, marker= '^', color='blue', label = 'Infubinol')
ax3 = drug_data['Ketapril'].plot(kind='line',yerr = sem_pivot['Ketapril'], linewidth=1, marker= 's', color='green', label='Ketapril')
ax4 = drug_data['Placebo'].plot(kind='line',yerr = sem_pivot['Placebo'], linewidth=1, marker= 'v', color='black', label='Placebo')
plt.legend(loc='upper left')
plt.title("Tumor Response to Treatment")
plt.xlabel("Time (Days)")
plt.ylabel("Tumor Volume (mm3)")
plt.axes().yaxis.grid()

# Save the Figure
plt.savefig("../Images/treatment.png")

# Show the chart
plt.show()
```


![png](output_6_0.png)


## Metastatic Response to Treatment


```python
# Store the Mean Met. Site Data Grouped by Drug and Timepoint 

met_data = combined_trial_df[['Drug', 'Timepoint', 'Metastatic Sites']]

# Convert to DataFrame

mean_met_data=met_data.groupby(['Drug','Timepoint'])['Metastatic Sites'].mean()
met_response_mean_data = pd.DataFrame(mean_met_data)

# Preview DataFrame

met_response_mean_data.head()

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
      <th></th>
      <th>Metastatic Sites</th>
    </tr>
    <tr>
      <th>Drug</th>
      <th>Timepoint</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="5" valign="top">Capomulin</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.160000</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.320000</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.375000</td>
    </tr>
    <tr>
      <td>20</td>
      <td>0.652174</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Store the Standard Error associated with Met. Sites Grouped by Drug and Timepoint 
met_error_data = combined_trial_df[['Drug', 'Timepoint', 'Metastatic Sites']]

# Convert to DataFrame

error_data = met_error_data.groupby(['Drug','Timepoint'])['Metastatic Sites'].sem()
met_error_data_df = pd.DataFrame(error_data)

# Preview DataFrame
met_error_data_df.head()
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
      <th></th>
      <th>Metastatic Sites</th>
    </tr>
    <tr>
      <th>Drug</th>
      <th>Timepoint</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="5" valign="top">Capomulin</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.074833</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.125433</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.132048</td>
    </tr>
    <tr>
      <td>20</td>
      <td>0.161621</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Minor Data Munging to Re-Format the Data Frames
met_mean_raws_to_column = met_response_mean_data.pivot_table('Metastatic Sites', ['Timepoint'], 'Drug')

# Preview that Reformatting worked
met_mean_raws_to_column.head()

# Preview that Reformatting worked

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
      <th>Drug</th>
      <th>Capomulin</th>
      <th>Ceftamin</th>
      <th>Infubinol</th>
      <th>Ketapril</th>
      <th>Naftisol</th>
      <th>Placebo</th>
      <th>Propriva</th>
      <th>Ramicane</th>
      <th>Stelasyn</th>
      <th>Zoniferol</th>
    </tr>
    <tr>
      <th>Timepoint</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.160000</td>
      <td>0.380952</td>
      <td>0.280000</td>
      <td>0.304348</td>
      <td>0.260870</td>
      <td>0.375000</td>
      <td>0.320000</td>
      <td>0.120000</td>
      <td>0.240000</td>
      <td>0.166667</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.320000</td>
      <td>0.600000</td>
      <td>0.666667</td>
      <td>0.590909</td>
      <td>0.523810</td>
      <td>0.833333</td>
      <td>0.565217</td>
      <td>0.250000</td>
      <td>0.478261</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.375000</td>
      <td>0.789474</td>
      <td>0.904762</td>
      <td>0.842105</td>
      <td>0.857143</td>
      <td>1.250000</td>
      <td>0.764706</td>
      <td>0.333333</td>
      <td>0.782609</td>
      <td>0.809524</td>
    </tr>
    <tr>
      <td>20</td>
      <td>0.652174</td>
      <td>1.111111</td>
      <td>1.050000</td>
      <td>1.210526</td>
      <td>1.150000</td>
      <td>1.526316</td>
      <td>1.000000</td>
      <td>0.347826</td>
      <td>0.952381</td>
      <td>1.294118</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Minor Data Munging to Re-Format the Data Frames
met_error_raws_to_column = met_error_data_df.pivot_table('Metastatic Sites', ['Timepoint'], 'Drug')

# Preview that Reformatting worked
met_error_raws_to_column.head()

# Preview that Reformatting worked

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
      <th>Drug</th>
      <th>Capomulin</th>
      <th>Ceftamin</th>
      <th>Infubinol</th>
      <th>Ketapril</th>
      <th>Naftisol</th>
      <th>Placebo</th>
      <th>Propriva</th>
      <th>Ramicane</th>
      <th>Stelasyn</th>
      <th>Zoniferol</th>
    </tr>
    <tr>
      <th>Timepoint</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.074833</td>
      <td>0.108588</td>
      <td>0.091652</td>
      <td>0.098100</td>
      <td>0.093618</td>
      <td>0.100947</td>
      <td>0.095219</td>
      <td>0.066332</td>
      <td>0.087178</td>
      <td>0.077709</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.125433</td>
      <td>0.152177</td>
      <td>0.159364</td>
      <td>0.142018</td>
      <td>0.163577</td>
      <td>0.115261</td>
      <td>0.105690</td>
      <td>0.090289</td>
      <td>0.123672</td>
      <td>0.109109</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.132048</td>
      <td>0.180625</td>
      <td>0.194015</td>
      <td>0.191381</td>
      <td>0.158651</td>
      <td>0.190221</td>
      <td>0.136377</td>
      <td>0.115261</td>
      <td>0.153439</td>
      <td>0.111677</td>
    </tr>
    <tr>
      <td>20</td>
      <td>0.161621</td>
      <td>0.241034</td>
      <td>0.234801</td>
      <td>0.236680</td>
      <td>0.181731</td>
      <td>0.234064</td>
      <td>0.171499</td>
      <td>0.119430</td>
      <td>0.200905</td>
      <td>0.166378</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Generate the Plot (with Error Bars)
met_sem_pivot = met_error_raws_to_column[['Capomulin', 'Infubinol','Ketapril','Placebo']]
met_drug_data = met_mean_raws_to_column[['Capomulin', 'Infubinol','Ketapril','Placebo']]
ax1 = met_drug_data['Capomulin'].plot(kind='line', yerr = met_sem_pivot['Capomulin'], linewidth=1, marker= 'o', color='red', label ='Capomulin')
ax2 = met_drug_data['Infubinol'].plot(kind='line',yerr = met_sem_pivot['Infubinol'],linewidth=1, marker= '^', color='blue', label = 'Infubinol')
ax3 = met_drug_data['Ketapril'].plot(kind='line',yerr = met_sem_pivot['Ketapril'], linewidth=1, marker= 's', color='green', label='Ketapril')
ax4 = met_drug_data['Placebo'].plot(kind='line',yerr = met_sem_pivot['Placebo'], linewidth=1, marker= 'v', color='black', label='Placebo')


plt.legend(loc='best')
plt.title("Metastatic spred during Treatment")
plt.xlabel("Treatment Duration (Days)")
plt.ylabel("Met. sites")
plt.xlim(-2,48)
plt.axes().yaxis.grid()
# Save the Figure
plt.savefig("../Images/Metastat spred.png")
# Show the Figure
plt.show()
```


![png](output_12_0.png)



```python
# Store the Count of Mice Grouped by Drug and Timepoint (W can pass any metric)
survival_rates_column = combined_trial_df[['Drug','Timepoint','Mouse ID']]                                    
total_count_data =survival_rates_column.groupby(['Drug','Timepoint'], as_index = False)['Mouse ID'].count()

# Convert to DataFrame
survival_data_df = pd.DataFrame(total_count_data)
survival_data_df = survival_data_df.rename(columns={'Mouse ID' : 'Mouse Count'})
# Preview DataFrame
survival_data_df.head()
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
      <th>Drug</th>
      <th>Timepoint</th>
      <th>Mouse Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Capomulin</td>
      <td>0</td>
      <td>25</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Capomulin</td>
      <td>5</td>
      <td>25</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Capomulin</td>
      <td>10</td>
      <td>25</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Capomulin</td>
      <td>15</td>
      <td>24</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Capomulin</td>
      <td>20</td>
      <td>23</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Minor Data Munging to Re-Format the Data Frames
survival_data_pivot_table = survival_data_df.pivot_table('Mouse Count', ['Timepoint'], 'Drug')
# Preview the Data Frame
survival_data_pivot_table.head()
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
      <th>Drug</th>
      <th>Capomulin</th>
      <th>Ceftamin</th>
      <th>Infubinol</th>
      <th>Ketapril</th>
      <th>Naftisol</th>
      <th>Placebo</th>
      <th>Propriva</th>
      <th>Ramicane</th>
      <th>Stelasyn</th>
      <th>Zoniferol</th>
    </tr>
    <tr>
      <th>Timepoint</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>25</td>
      <td>25</td>
      <td>25</td>
      <td>25</td>
      <td>25</td>
      <td>25</td>
      <td>26</td>
      <td>25</td>
      <td>26</td>
      <td>25</td>
    </tr>
    <tr>
      <td>5</td>
      <td>25</td>
      <td>21</td>
      <td>25</td>
      <td>23</td>
      <td>23</td>
      <td>24</td>
      <td>25</td>
      <td>25</td>
      <td>25</td>
      <td>24</td>
    </tr>
    <tr>
      <td>10</td>
      <td>25</td>
      <td>20</td>
      <td>21</td>
      <td>22</td>
      <td>21</td>
      <td>24</td>
      <td>23</td>
      <td>24</td>
      <td>23</td>
      <td>22</td>
    </tr>
    <tr>
      <td>15</td>
      <td>24</td>
      <td>19</td>
      <td>21</td>
      <td>19</td>
      <td>21</td>
      <td>20</td>
      <td>17</td>
      <td>24</td>
      <td>23</td>
      <td>21</td>
    </tr>
    <tr>
      <td>20</td>
      <td>23</td>
      <td>18</td>
      <td>20</td>
      <td>19</td>
      <td>20</td>
      <td>19</td>
      <td>17</td>
      <td>23</td>
      <td>21</td>
      <td>17</td>
    </tr>
  </tbody>
</table>
</div>




```python

# Generate the Plot (Accounting for percentages)
selected_drug_data = survival_data_df.pivot_table('Mouse Count', ['Timepoint'], 'Drug')[['Capomulin', 'Infubinol','Ketapril','Placebo']]
selected_drug_data =selected_drug_data.div(selected_drug_data.iloc[0,0]).multiply(100)
# Preview the Data Frame
selected_drug_data
# Generate the Plot
selected_drug_data.plot(kind ='line', style = ['ro-','b^-','gs-','kv-'])

plt.legend(loc='best')
plt.title("Survival during Treatment")
plt.xlabel("Time (Days)")

plt.ylabel("Survival rate(%)")
plt.xlim(-3,50)
plt.ylim(30,110)
plt.grid()

# Save the Figure
plt.savefig("../Images/Survival Rate.png")
# Show the Figure
plt.show()
```


![png](output_15_0.png)



## Summary Bar Graph


```python
# Calculate the percent changes for each drug

trial_mean_data =combined_trial_df.groupby(['Drug','Timepoint'])['Tumor Volume (mm3)'].mean()
trial_mean_data

# Preview DataFrame

def first_last(df):
    return df.ix[[0, -1]].pct_change().iloc[[-1]].multiply(100)

percent_change_data = trial_mean_data.groupby(level=0, group_keys=False).apply(first_last).reset_index()

percent_change_df = percent_change_data.drop(['Timepoint'], axis = 1)
percent_change_df
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
      <th>Drug</th>
      <th>Tumor Volume (mm3)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Capomulin</td>
      <td>-19.475303</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Ceftamin</td>
      <td>42.516492</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Infubinol</td>
      <td>46.123472</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Ketapril</td>
      <td>57.028795</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Naftisol</td>
      <td>53.923347</td>
    </tr>
    <tr>
      <td>5</td>
      <td>Placebo</td>
      <td>51.297960</td>
    </tr>
    <tr>
      <td>6</td>
      <td>Propriva</td>
      <td>47.241175</td>
    </tr>
    <tr>
      <td>7</td>
      <td>Ramicane</td>
      <td>-22.320900</td>
    </tr>
    <tr>
      <td>8</td>
      <td>Stelasyn</td>
      <td>52.085134</td>
    </tr>
    <tr>
      <td>9</td>
      <td>Zoniferol</td>
      <td>46.579751</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Store all Relevant Percent Changes into a Tuple

selected_drug = ['Capomulin', 'Infubinol','Ketapril','Placebo']
tuples = [tuple(x) for x in percent_change_df.values if x[0] in selected_drug]

# Orient widths. Add labels, tick marks, etc. 

def select_color(percentage):
    if percentage < 0:
        return 'g'
    else:
        return 'r'

plt.bar(selected_drug, 
        [x[1] for x in tuples],
        align ='center', width=1.0,
        color= [select_color(x[1]) for x in tuples],
        alpha=1.0)

plt.grid()
plt.title("Survival during Treatment")
plt.ylabel("% tumor volume change")
plt.title("tumor change over 45 days")

plt.xticks(np.arange(0.5, 4, step=0.999))


# Use functions to label the percentages of changes

plt.xlim(-0.7,3.7)
plt.ylim(-30,70)

def label_position(percentage):
    if percentage < 0:
        return -5
    else:
        return 5



# Call functions to implement the function calls

for index, (drug, percentage) in enumerate(tuples):
    plt.text(index,
             label_position(percentage),
             str(round(percentage,0)) + '%',
             color='w', fontweight='bold',ha='center', va='center')

# Save the Figure
plt.savefig("../Images/change.png")

# Show the Figure

plt.show()




```


![png](output_18_0.png)



```python

```
