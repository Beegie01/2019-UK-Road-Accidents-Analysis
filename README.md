# 2019-UK-Road-Accidents-Data-Mining
This is my analysis of 2019 road accidents in the UK. Data (downloaded from the official Uk government website gov.uk) were in separate accidents, vehicles, and casualties CSV files, as well as variable lookup XLS files. The datasets were merged on their common identifier, ‘accident_index’, and the result was a table having 295,579 rows and 69 columns. <br><br>
My goal was to extract sufficient knowledge from the data through analyses and use this knowledge to build a predictive model that can assist government officials in making precautionary policies to prevent enabling conditions for road accidents.<br>
The project consisted of the following subtasks: data cleaning, feature engineering, correlation analysis, association mining, inferential statistics, and recommendation.<br><br>
### i. DATA CLEANING:<br>
[missing_entries.png](https://user-images.githubusercontent.com/76821049/173702745-a483ae4f-b175-4de5-99b2-215d1e23a97f.png)<br>
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
Correlated features are those with absolute Pearson's correlation values ≥ 0.5 (50%)<br>
[correlated_features.png](https://user-images.githubusercontent.com/76821049/173705642-ac9c061e-73a0-40ce-aade-d70ab2aec5db.png)<br><br>

### iv. ANALYSIS & INFERENTIAL STATISTICS<br>
Statistical testing was performed using a pooled variance. Statistical significance was established when the mean_difference ≥ 2 x pooled_std (95% confidence), or 3 x pooled_std (99% confidence).<br><br>

#### Overall Observation<br>
[Total_hourly_accident_distribution.png](https://user-images.githubusercontent.com/76821049/173780394-dcdd3df2-a5e9-414a-b6a5-235c062cc562.png)<br><br>

#### AM hourly accidents VS. PM hourly accidents<br>
Statistical test results suggest that sunlight (brightness of day) which promotes human activity in general, increased the likelihood of an accident.<br>
[Here](https://user-images.githubusercontent.com/76821049/173707051-c74f17eb-171f-4b48-a86b-5a15ffc69e4e.png) is an illustration of the hourly difference between AM and PM accidents<br><br>

#### Daily road accidents<br>
Statistical test results suggest that overall accidents were more likely to occur on the weekday than the weekend probably because more people commute on weekdays (e.g to schools, banks).<br>
[Here](https://user-images.githubusercontent.com/76821049/173796074-c541b04e-8835-4e45-9932-0528044f9bd7.png) is an illustration of overall weekday accidents<br>
[Here](https://user-images.githubusercontent.com/76821049/173796298-c5ed0f74-f222-4225-85ad-8aa4e56b044d.png) is an illustration of the seasonal weekday accidents<br><br>

Statistical test results suggest that nighttime accidents were more likely to occur on a Friday-Saturday-Sunday than a Monday-Tuesday-Wednesday-Thursday.<br>
[Here](https://user-images.githubusercontent.com/76821049/173777508-fc7ce2e2-fdb6-4016-948f-45d139d34681.png) is an illustration of weekday accidents per part of day<br><br>

#### Pedestrian Accidents<br>
[Pedestrian_hourly_accidents.png](https://user-images.githubusercontent.com/76821049/173781853-d62304e3-b029-45a2-9c19-ae8fe1907ef0.png)<br><br>

#### Effect of Daylight Saving Time (DST) on road accidents<br>
Statistical test results suggest that onset of DST caused a decrease in 9am – 11.59am accidents.<br>
[This](https://user-images.githubusercontent.com/76821049/173798168-750b2e7a-6c90-474c-b84a-29f8c3ccca04.png) illustrates the difference between hourly accidents that occurred during the FIRST week of DST vs. the week BEFORE<br>

[This](https://user-images.githubusercontent.com/76821049/173798742-cc475449-c5d3-4a01-8fea-5866d1fcb538.png) illustrates the difference between hourly accidents that occurred during the LAST week of DST vs. the week AFTER<br><br>

#### Impact of Sunrise and Sunset on road accidents<br>
Statistical test results suggest that sunrise caused an increase in accidents at 7am – 9.59am.<br>
[This diagram](https://user-images.githubusercontent.com/76821049/173785277-f179d886-f070-4186-8da9-f08f8e2e6da9.png) shows the difference between accidents within 2 hours BEFORE vs. 2 hours AFTER sunrise<br>
Statistical test results suggest that sunset caused a decline in accidents at 14pm – 15.59pm.<br>
[This diagram](https://user-images.githubusercontent.com/76821049/173785534-3c6e38fc-b909-427a-bd27-998e7c51ee5c.png) shows the difference between accidents within 2 hours BEFORE vs. 2 hours AFTER sunset<br><br>

#### Influence of Vehicular Variables on road accidents<br>
Below diagrams illustrate that the most accident-prone vehicles were 7 year-old cars, vans, and pedal cycles having 124cc engine-capacity moving ahead on an unrestricted main carriageway in a 30mph speed limit zone.<br>
[Association_between_vehicle_type, vehicle_age, engine_capacity, and propulsion_code.png](https://user-images.githubusercontent.com/76821049/173786685-22af2678-2099-4156-8947-4297654b85ef.png)<br>
[Association_between_vehicle_type, speed_limit, and vehicle propulsion.png](https://user-images.githubusercontent.com/76821049/173791164-c1f78588-66d0-4553-b782-e559ff732022.png)<br>
[Association_between_vehicle_location_restricted_lane, vehicle_manoeuvre, propulsion_code.png](https://user-images.githubusercontent.com/76821049/173851789-057ca721-7dbc-4410-84bf-e0a3fd27ef39.png)<br><br>

#### Influence of Weather on road accidents<br>
[Here,](https://user-images.githubusercontent.com/76821049/173803895-14ca0506-f85c-407a-b87c-8f4a464f55a0.png) it is revealed that all year round most accidents happenned on a dry road surface under fine weather conditions with no high wind, **except the autumn** during which most accidents happened on a wet/damp road surface when it rained with no high wind.<br><br>

#### Influence of geographic variables<br>
[Here](https://user-images.githubusercontent.com/76821049/173845340-3d4a537e-7e0e-40ef-97a1-612bda4a7481.png) are the **top ten UK districts** with the highest number of accidents in 2019. Majority of the top ten districts are located in the **central region**<br>
[Here](https://user-images.githubusercontent.com/76821049/173845893-f67a1e77-e860-4fdc-af72-71ef90934193.png) are the **top ten UK highways** with highest number of accidents in 2019. Most of the top ten highways are located in the **eastern region**<br>

