# 2019-UK-Road-Accidents-Data-Mining
This is my analysis of 2019 road accidents in the UK. Data (downloaded from the official Uk government website gov.uk) were in separate accidents, vehicles, and casualties CSV files, as well as variable lookup XLS files. The datasets were merged on their common identifier, ‘accident_index’, and the result was a table having 295,579 rows and 69 columns. <br><br>
My goal was to extract sufficient knowledge from the data through analyses and use this knowledge to build a predictive model that can assist government officials in making precautionary policies to prevent enabling conditions for road accidents.<br>
The project consisted of the following subtasks: data cleaning, feature engineering, correlation analysis, association mining, inferential statistics, and recommendation.<br><br>
### i. DATA CLEANING:<br>
![image](https://user-images.githubusercontent.com/76821049/173702745-a483ae4f-b175-4de5-99b2-215d1e23a97f.png)<br>
After dropping the rows for features missing less than 2.5% of total number of rows, the remaining missing rows were replaced with the modal values per unique category of the top 2 correlated features. Finally, 93% of the data was retained (after cleaning) for further analysis and prediction.<br><br>

### ii.	FEATURE ENGINEERING<br>
Engineered features include:<br>
1.	part_of_day: <br>
  1 --> morning (5am – 11.59am)<br>
  2 --> afternoon (12pm – 16.59pm)<br> 
  3 --> evening (17 – 20.59pm)<br>
  4 --> night (21pm – 4.59am)<br>
2.	is_weekend: Weekend is Saturday and Sunday.
3.	quarter: yearly quarters
4.	season: (autumn, spring, summer, and winter).
5.	is_dst (2019 Daylight Saving Time): 1am March 31 to 2am October 27.
6.	is_offseason (Premier League offseason indicator): 13 May – 8 August 2019
7.	is_sunrise and is_sunset: sunrise/sunset time boundaries.<br>
The time and date columns were dropped to minimize information duplicity. <br><br>

### iii. CORRELATION ANALYSIS<br>
Correlated features are those with absolute Pearson's correlation values ≥ 0.5 (50%)
![image](https://user-images.githubusercontent.com/76821049/173705642-ac9c061e-73a0-40ce-aade-d70ab2aec5db.png)<br><br>

### iv. INFERENTIAL STATISTICS<br>
Statistical testing was performed using a pooled variance. Statistical significance was established when the mean_difference ≥ 2 x pooled_std (95% confidence), or 3 x pooled_std (99% confidence).<br><br>
#### Overall Observation<br>
![image](https://user-images.githubusercontent.com/76821049/173707051-c74f17eb-171f-4b48-a86b-5a15ffc69e4e.png)<br>
Statistical test results suggest that sunlight (brightness of day) which promotes human activity in general, increased the likelihood of an accident.<br>
![image](https://user-images.githubusercontent.com/76821049/173707403-288fadad-58b4-4aa5-95a0-a34424164a5d.png)<br>
Statistical test results suggest that overall accidents were more likely to occur on the weekday than the weekend probably because more people commute on weekdays (e.g to schools, banks).<br><br>
![image](https://user-images.githubusercontent.com/76821049/173777508-fc7ce2e2-fdb6-4016-948f-45d139d34681.png)<br>
Statistical test results suggest that nighttime accidents were more likely to occur on a Friday-Saturday-Sunday than a Monday-Tuesday-Wednesday-Thursday.
