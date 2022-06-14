# 2019-UK-Road-Accidents-Analysis
This is my analysis of 2019 road accidents in the UK. Data (downloaded from the official Uk government website gov.uk) were in separate accidents, vehicles, and casualties CSV files, as well as variable lookup XLS files. The datasets were merged on their common identifier, ‘accident_index’, and the result was a table having 295,579 rows and 69 columns. <br>
My goal was to extract sufficient knowledge from the data through analyses and use this knowledge to build a predictive model that can assist government officials in making precautionary policies to prevent enabling conditions for road accidents.<br>
The project consisted of the following subtasks: data cleaning, feature engineering, correlation analysis, association mining, inferential statistics, and recommendation.<br><br>
DATA CLEANING:<br>
![image](https://user-images.githubusercontent.com/76821049/173702745-a483ae4f-b175-4de5-99b2-215d1e23a97f.png)<br>
After dropping the missing rows for variables missing less than 2.5% of total number of rows, the remaining missing rows were replaced with modal values for each unique category of their top 2 correlated variables.<br>
![image](https://user-images.githubusercontent.com/76821049/173704301-f7057d6f-b6d0-4cd6-a8c3-e96fd14be0fc.png)<br>
Retained 93% of the data (after cleaning) for further analysis and prediction.
