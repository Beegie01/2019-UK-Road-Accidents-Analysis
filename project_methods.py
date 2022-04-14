import string
import copy
import seaborn as sns
from seaborn.utils import os, np, plt, pd
from sklearn import preprocessing as s_prep, metrics as s_mtr, feature_selection as s_fs, pipeline as s_pipe
from tensorflow.keras import models, layers, backend as K, callbacks
from scipy import stats


class RoadAccidents:
    
    def __init__(self):
        os.makedirs(self.app_folder_loc, exist_ok=True)  # create folder 'Files' at current working directory for storing useful files
    
    @property
    def app_name(self):
        return 'RoadAccidents'
    
    @property
    def app_folder_loc(self):
        return f'{self.app_name}Output'
        
    @staticmethod
    def compute_pvalue(x1, x2, h1, seed):
        """compute test_statistic and pvalue of null hypothesis.
        h1: {'greater', 'less', 'two-sided'}
        Returns:
        statistic : float or array
            The calculated t-statistic.
        pvalue : float or array
            The two-tailed p-value."""
    
    #     print(help(stats.ttest_ind))
        np.random.seed(seed)
        
        tstat, pval = np.round(stats.ttest_ind(x1, x2, equal_var=False, alternative=h1), 4)
        return tstat, pval
        
    @staticmethod
    def assign_seasons(df):
        """generate 2019 UK seasons from month and day columns using.
        spring: march 20 - june 20
        summer: june 21 - september 22
        autumn: september 23 - december 20
        winter: december 21 - march 19
        
        steps
        1. select full months
        2. exclude outside days
        Return:
        seasons: series in string format"""
       
        seasons = {'spring_months': (4, 5),
                  'summer_months': (7, 8),
                  'autumn_months': (10, 11),
                  'winter_months': (1, 2)}
        
        # select index of spring months
        full_months = (df['month'].isin(seasons['spring_months']))
        exclude_march_days = ((df['month'] == 3)  &
                               (df['day'] >= 20))
        exclude_june_days = ((df['month'] == 6)  &
                              (df['day'] <= 20))
        
        spring_ind = df.loc[full_months|
                             exclude_march_days |
                             exclude_june_days].index
        
        # select index of summer months
        full_months = (df['month'].isin(seasons['summer_months']))
        exclude_june2_days = ((df['month'] == 6)  &
                               (df['day'] >= 21))
        exclude_sept_days = ((df['month'] == 9)  &
                              (df['day'] <= 22))
        
        summer_ind = df.loc[full_months|
                             exclude_june2_days |
                             exclude_sept_days].index
        
        # select index of autum months
        full_months = (df['month'].isin(seasons['autumn_months']))
        exclude_sept2_days = ((df['month'] == 9)  &
                               (df['day'] >= 23))
        exclude_dec_days = ((df['month'] == 12)  &
                              (df['day'] <= 20))
        
        autumn_ind = df.loc[full_months|
                             exclude_sept2_days |
                             exclude_dec_days].index
        
        # select index of winter months
        full_months = (df['month'].isin(seasons['winter_months']))
        exclude_dec2_days = ((df['month'] == 12)  &
                               (df['day'] >= 21))
        exclude_march2_days = ((df['month'] == 3)  &
                              (df['day'] <= 19))
        
        winter_ind = df.loc[full_months|
                             exclude_dec2_days |
                             exclude_march2_days].index
        
        # assign seasons to empty series
        result = pd.Series(index=df.index, dtype='object')
        result.loc[spring_ind] = 'spring'
        result.loc[summer_ind] = 'summer'
        result.loc[autumn_ind] = 'autumn'
        result.loc[winter_ind] = 'winter'
        return result
        
    @staticmethod
    def compute_stat_pval(x1, x2, h1='greater',  X1_name='X1', X2_name='X2', deg_freedom=1, seed=1):
        """report statistical significance using pvalue"""
    
        
        def get_min_denom(n1, n2):
                return min([n1, n2])
            
        def detect_unequal_sizes(n1, n2):
            """check if both lengths are not the same"""
            return n1 != n2
        
        cls = DsUtils()

        np.random.seed(seed)
        X1_size, X2_size = len(x1), len(x2)
        samp_sizes = {X1_name: pd.Series(x1),
                      X2_name: pd.Series(x2)}
        print(f'ORIGINAL SAMPLE SIZE: \n{X1_name}: {X1_size}\n' +
              f'{X2_name}: {X2_size}\n\n')

        # check if sample sizes are unequal
        if detect_unequal_sizes(X1_size, X2_size):
            print("Unequal Sample Sizes Detected!!")
            min_size = get_min_denom(X1_size, X2_size)
            max_samp_name = [name for name, val in samp_sizes.items() if len(val) != min_size][0]
            # downsampling: 
            # randomly generate min_size indexes for max_samp_name
            rand_indexes = cls.index_generator(len(samp_sizes[max_samp_name]), min_size)
            # select only random min_size indexes for max_samp_name set
            samp_sizes[max_samp_name] = samp_sizes[max_samp_name].iloc[rand_indexes]
            X1_size, X2_size = len(samp_sizes[X1_name]), len(samp_sizes[X2_name])
            print(f'ADJUSTED SAMPLE SIZE: \n{X1_name}: {X1_size}\n' +
                  f'{X2_name}: {X2_size}\n\n')
            
        total_X1, X1_mean, X1_std = cls.compute_mean_std(samp_sizes[X1_name], X1_size, deg_freedom)
        total_X2, X2_mean, X2_std = cls.compute_mean_std(samp_sizes[X2_name], X2_size, deg_freedom)

        null_hypo = np.round(X1_mean - X2_mean, 4)
        pooled_std = cls.compute_pstd(X1_std, X2_std)

        print(f'{X1_name}:\n Total = {total_X1}\n Average = {X1_mean}\n Standard deviation = {X1_std}\n\n' +
              f'{X2_name}:\n Total = {total_X2}\n Average = {X2_mean}\n Standard deviation = {X2_std}\n\n' +
              f'MEAN DIFFERENCE = {null_hypo}\n' +
              f'POOLED STD = {pooled_std}\n\n')
        
        tstat, pval = cls.compute_pvalue(samp_sizes[X1_name], samp_sizes[X2_name], seed)
        test_result = (['99%', 0.01], 
                       ['95%', 0.05])
        if pval <= test_result[0][1]:
            return print(f'At {test_result[0][0]}, REJECT the null hypothesis!\n {pval} is less than {test_result[0][1]}\n')
        elif pval <= test_result[1][1]:
                return print(f'At {test_result[1][0]}, REJECT the null hypothesis!\n{pval} is less than or equal to {test_result[1][1]}\n')
        print(f'Do NOT reject the null hypothesis\n{pval} is greater than {test_result[1][1]}')
        
    @staticmethod
    def aggregated_trend(database, fig_filename_suffix='fig'):
    
        cls = RoadAccidents()
        general_agg = cls.generate_aggregated_lookup(database)
        # 1. average hourly accident per day of week showing weekend and weekdays
        cols = ['hour', 'day_name', 'is_weekend', 'total_count']
        acc_hr_wkend = general_agg[cols].groupby(cols[:-1]).mean().reset_index()
        #print(acc_hr_wkend)

        color_mapping = {0: 'red', 1: 'gray'}

        fig, ax = plt.subplots(1, 1, figsize=(18, 8), dpi=150)
        cls.plot_column(acc_hr_wkend['hour'], acc_hr_wkend['total_count'], annotate=False, condition_on=acc_hr_wkend['is_weekend'],
                        paletter=color_mapping, plot_title=f'{str.capitalize(fig_filename_suffix)} Weekend Vs. Weekday Average Accidents per Hour', axis=ax)
        line1 = "From 22pm to 4am and 10am to 13pm, there were more accidents on weekends.\n"
        line2 = "From 5am to 9am and 14pm to 19pm, there were more accidents on weekdays."
        annotation = line1+line2
        ax.text(0, 65, annotation, color='black', weight='bold', size=12,
               bbox={'facecolor': 'none', 'edgecolor':'blue'})
        fig_filename = f'wkend_distribution_{fig_filename_suffix.lower()}.png'
        print(cls.fig_writer(fig_filename, fig))

        # 2. total accidents per day of week
        cols = ['hour', 'day_of_week', 'day_name', 'total_count']
        acc_hr_dayname = general_agg[cols].groupby(cols[:-1]).sum().reset_index()
        acc_hr_dayname

        color_mapping = {'Sunday': 'green', 'Monday': 'blue', 'Tuesday': 'yellow', 'Wednesday': 'darkorange', 'Thursday': 'red', 'Friday': 'black', 'Saturday': 'gray'}

        fig, ax = plt.subplots(1, 1, figsize=(18, 8), dpi=150)
        cls.plot_column(acc_hr_dayname['hour'], acc_hr_dayname['total_count'], annotate=False, condition_on=acc_hr_dayname['day_name'],
                        paletter=color_mapping, axis=ax, plot_title='Total Hourly Accidents per Day of Week')
        line1 = "From 0am to 4am, highest accidents were on Saturdays & Sundays.\n"
        line2 = "From 6am to 9am, higest accidents were on Tuesdays, Wednesdays and Thursdays.\n"
        line3 = "From 10am to 14pm and 19pm to 23pm, accidents were highest on Fridays & Saturdays.\n"
        line4 = "From 15pm to 18pm, accidents were highest on Tuesdays, Wednesdays, Thursdays, and Fridays."
        annotation = line1+line2+line3+line4
        ax.text(0, 4000, annotation, color='black', weight='bold', size=13,
               bbox={'facecolor': 'none', 'edgecolor':'blue'})
        ax.set_ylim(top=5000)
        fig_filename = f'dayname_distribution_{fig_filename_suffix.lower()}.png'
        print(cls.fig_writer(fig_filename, fig))
        
    @staticmethod
    def show_correlated_variables(df, var_look):
        
        
        cls = RoadAccidents()
        # 1. show relationship between accident_severity vs casualty_severity
        # plot column
        x, y = 'accident_severity', 'casualty_severity'
        freq = df[[x, y]].value_counts().sort_index().reset_index()
        freq.columns = freq.columns.astype(str).str.replace('0', 'total_count')
        freq['%total'] = cls.give_percentage(freq['total_count'])
        ll = dict(var_look['Casualty Severity'].to_records(index=False))
        freq[x] = freq.apply(lambda row: ll[row[x]],
                  axis=1)
        freq[y] = freq.apply(lambda row: ll[row[y]],
                  axis=1)
        # print(freq)

        fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=150)
        cls.plot_column(freq[x], freq['total_count'], condition_on=freq[y],
                    annotate=False, paletter={'Fatal': 'red', 'Serious': 'orange', 'Slight': 'gray'}, axis=ax, 
                    plot_title='Relationship Between Accident and Casualty Severity')

        show_table = f"{freq.drop('total_count', axis=1).to_string()}"
        ax.text(0.75, 80000, show_table, fontsize=8,  weight='bold',
                bbox={'edgecolor':'red', 'facecolor':'none'})

        line1 = 'Majority of accidents are SLIGHTLY severe.\nMajority of casualties sustained SLIGHT injuries.'  
        line2 = '\nMajority (1.14%) of casualties in FATAL accidents incurred FATAL injuries.'
        line3 = '\nMajority (16%) of casualties in SERIOUS accidents incurred SERIOUS injuries'
        line4 = '\nMajority (77%) of casualties in SLIGHT accidents incurred SLIGHT injuries'
        annot = line1+line2+line3+line4
        ax.text(-0.35, 160000, annot, color='brown',# weight='bold',
                 bbox={'edgecolor':'red', 'facecolor':'none'})
        sns.move_legend(ax, [0.05, 0.35])
        fname = 'hist_acc_cas_severity.png'
        plt.show()
        print(cls.fig_writer(fname, fig));

        # plot scatter
        fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=150)

        cls.plot_scatter(df[x], df[y], 
                          plot_title='Relationship Between Accident and Casualty Severity', axis=ax)

        line1 = "In SLIGHTLY severe accidents, casualties sustained only minor injuries.\n"
        line2 = "In SERIOUS accidents, casualties sustained both serious and minor injuries\n"
        line3 = "In FATAL accidents, casualties sustained slight, serious, and fatal outcomes"
        scatter_msg = line1+line2+line3
        ax.text(1.20, 1.15, scatter_msg, color='black', bbox={'edgecolor':'red', 'facecolor':'none'})
        legnd = var_look['Casualty Severity'].to_string(index=False)
        # print(annot)
        ax.text(2.65, 2.45, legnd, color='blue', weight='bold', bbox={'edgecolor':'red', 'facecolor':'none'})
        fname = 'scatt_acc_cas_severity.png'
        plt.show()
        print(cls.fig_writer(fname, fig));
        
        # 2a. show relationship betweeen vehicle_reference_x and number of vehicles
        x, y = 'vehicle_reference_x', 'number_of_vehicles'
        fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=150)
        cls.plot_line(df[x], df[y], marker='o',
                    axis=ax, plot_title=f'Relationship Between {x.capitalize()} and {y.capitalize()}')
        
        line1 = f'In most cases, as {x} increased in magnitude, \nso too did {y}.'
        ax.text(0.5, 14, line1, color='blue', size=7,# weight='bold',
                 bbox={'edgecolor':'red', 'facecolor':'none'})

        plt.show()
        fname = 'line_veh_num_refx.png'
        plt.show()
        print(cls.fig_writer(fname, fig));
        
        # 2b. show relationship betweeen vehicle_reference_y and number of vehicles
        x, y = 'vehicle_reference_y', 'number_of_vehicles'
        fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=150)
        cls.plot_line(df[x], df[y], marker='o',
                    axis=ax, plot_title=f'Relationship Between {x.capitalize()} and {y.capitalize()}')
        
        line1 = f'In most cases, as {x} increased in magnitude, \nso also did {y}.'
        ax.text(0.5, 14, line1, color='blue', size=7,# weight='bold',
                 bbox={'edgecolor':'red', 'facecolor':'none'})
        plt.show()
        fname = 'line_veh_num_refy.png'
        plt.show()
        print(cls.fig_writer(fname, fig));
        
        # 3a. show relationship betweeen speed_limit and urban_or_rural_area
        x, y = 'speed_limit', 'urban_or_rural_area'
        fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=150)
        cls.plot_line(df[x], df[y], marker='o',
                    axis=ax, plot_title=f'Relationship Between {x.capitalize()} and {y.capitalize()}')
        line1 = f'There is a positive relationship between {x} and {y}.\n'
        line2 = f'In most cases, when the value of {x} was high, \nso was the value of {y}.'
        ax.text(41, 1.3, line1+line2, color='black', size=6, weight='bold',
                 bbox={'edgecolor':'red', 'facecolor':'none'})
        plt.show()
        fname = 'line_spd_urb.png'
        plt.show()
        print(cls.fig_writer(fname, fig));
        
        # 3b. column plot
        freq = df[[x, y]].value_counts().sort_index().reset_index()
        freq.columns = freq.columns.astype(str).str.replace('0', 'total_count')
        ll = dict(var_look['Speed Limit'].to_records(index=False))
        bb = dict(var_look['Urban Rural'].to_records(index=False))
        freq[x] = freq.apply(lambda row: ll[row[x]],
                  axis=1)
        freq[y] = freq.apply(lambda row: bb[row[y]],
                  axis=1)
        freq['%total'] = cls.give_percentage(freq['total_count'])
    #     print(freq)

        fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=150)
        cls.plot_column(freq[x], freq['%total'], freq[y], annotate=False, 
                         axis=ax, plot_title=f'Relationship Between {x.capitalize()} and {y.capitalize()}',
                        paletter={'Urban': 'green', 'Rural': 'gray'})

        show_table = f"{freq.drop('total_count', axis=1).to_string()}"
        ax.text(1.5, 8, show_table, fontsize=8,  weight='bold',
                bbox={'edgecolor':'red', 'facecolor':'none'})

        sns.move_legend(ax, [0.75, 0.35])

        line1 = f'The relationship between {x} and {y} is straightforward.\n'
        line2 = 'Speed limit is higher in rural areas'
        annot = line1+line2
        ax.text(1.15, 37, annot, color='blue', size=8, weight='bold',
                 bbox={'edgecolor':'red', 'facecolor':'none'})
        
        fname = 'hist_spd_urb.png'
        plt.show()
        print(cls.fig_writer(fname, fig));
        
        # 4. show relationship betweeen driver_home_area and casualty_home_are
        x, y = 'driver_home_area_type', 'casualty_home_area_type'
        # plot column
        freq = df[[x, y]].value_counts().sort_index().reset_index()
        freq.columns = freq.columns.astype(str).str.replace('0', 'total_count')
        ll = dict(var_look['Home Area Type'].to_records(index=False))
        freq[x] = freq.apply(lambda row: ll[row[x]],
                  axis=1)
        freq[y] = freq.apply(lambda row: ll[row[y]],
                  axis=1)
        freq['%total'] = cls.give_percentage(freq['total_count'])
        # print(freq)

        fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=150)
        cls.plot_column(freq[x], freq['%total'], freq[y], annotate=False,
                    axis=ax, plot_title=f'Relationship Between {x.capitalize()} and {y.capitalize()}',
                        paletter={'Urban area': 'green', 'Small town': 'yellow', 'Rural': 'gray'})

        show_table = f"{freq.drop('total_count', axis=1).to_string()}"
        ax.text(0, 15, show_table, fontsize=8,  weight='bold',
                bbox={'edgecolor':'red', 'facecolor':'none'})

        line1 = f'When {x} was urban, home area of majority (76%) of casualties was also urban.\n'
        line2 = f'When {x} was a small town, home area of majority (5.4%) of casualties was also small town.\n'
        line3 = f'When {x} was rural, home area of majority (7%) of casualties was also rural.'

        annot = line1+line2+line3
        ax.text(-0.08, 60, annot, color='blue', size=7, weight='bold',
                 bbox={'edgecolor':'red', 'facecolor':'none'})
        sns.move_legend(ax, [0.7, 0.35])
        
        fname = 'hist_dr_cas_home.png'
        plt.show()
        print(cls.fig_writer(fname, fig));

        # 5a.show relationship betweeen casualty class and pedestrian_location
        x, y = 'casualty_class', 'pedestrian_location'

        freq = df[[x, y]].value_counts().sort_index().reset_index()
        freq.columns = freq.columns.astype(str).str.replace('0', 'total_count')
        ll = dict(var_look['Casualty Class'].to_records(index=False))
        bb = dict(var_look['Ped Location'].to_records(index=False))
        freq[x] = freq.apply(lambda row: ll[row[x]],
                  axis=1)
        freq[y] = freq.apply(lambda row: bb[row[y]],
                  axis=1)
        freq['%total'] = cls.give_percentage(freq['total_count'])
        # print(freq)

        fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=150)
        cls.plot_column(freq[x], freq['%total'], freq[y], annotate=False, 
                         axis=ax, plot_title=f'Relationship Between {x.capitalize()} and {y.capitalize()}',)
        #                 paletter={'Urban': 'green', 'Rural': 'gray'})

        show_table = f"{freq.drop('total_count', axis=1).to_string()}"
        ax.text(-0.25, 26, show_table, fontsize=8,  weight='bold',
                bbox={'edgecolor':'red', 'facecolor':'none'})

        sns.move_legend(ax, [1.025, 0.15])

        line1 = f'Majority of the casualties are not pedestrians.'
        annot = line1
        ax.text(1.15, 10, annot, color='blue', size=8, weight='bold',
                 bbox={'edgecolor':'red', 'facecolor':'none'})
        fname = 'hist_cascls_pedloc.png'
        plt.show()
        print(cls.fig_writer(fname, fig));
        
        
        # 5b.show relationship betweeen casualty class and pedestrian_movement
        x, y = 'casualty_class', 'pedestrian_movement'

        freq = df[[x, y]].value_counts().sort_index().reset_index()
        freq.columns = freq.columns.astype(str).str.replace('0', 'total_count')
        ll = dict(var_look['Casualty Class'].to_records(index=False))
        bb = dict(var_look['Ped Location'].to_records(index=False))
        freq[x] = freq.apply(lambda row: ll[row[x]],
                  axis=1)
        freq[y] = freq.apply(lambda row: bb[row[y]],
                  axis=1)
        freq['%total'] = cls.give_percentage(freq['total_count'])
        # print(freq)

        fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=150)
        cls.plot_column(freq[x], freq['%total'], freq[y], annotate=False, 
                         axis=ax, plot_title=f'Relationship Between {x.capitalize()} and {y.capitalize()}',)
        #                 paletter={'Urban': 'green', 'Rural': 'gray'})

        show_table = f"{freq.drop('total_count', axis=1).to_string()}"
        ax.text(-0.25, 26, show_table, fontsize=8,  weight='bold',
                bbox={'edgecolor':'red', 'facecolor':'none'})

        sns.move_legend(ax, [1.025, 0.15])

        line1 = f'Majority of the casualties are not pedestrians.'
        annot = line1
        ax.text(1.15, 10, annot, color='blue', size=8, weight='bold',
                 bbox={'edgecolor':'red', 'facecolor':'none'})
        fname = 'hist_cascls_pedmovt.png'
        plt.show()
        print(cls.fig_writer(fname, fig));

        # 6. show relationship between pedestrian_location and pedestrian_movement
        x, y = 'pedestrian_location', 'pedestrian_movement'
        fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=150)
        cls.plot_line(df[y], df[x], marker='o',
                      plot_title=f'Relationship Between {x} and {y}', axis=ax)
        line1 = f'Non-straightforward relationship between variables.'
        annot = line1
        ax.text(1.15, 2.5, annot, color='black', size=8, weight='bold',
                 bbox={'edgecolor':'orange', 'facecolor':'none'})
        fname = 'scatt_pedloc_pedmovt.png'
        print(cls.fig_writer(fname, fig));
        
        # 7. month and season
        x, y = ['month', 'season']

        freq = df[[x, y]].value_counts().sort_index().reset_index()
        freq.columns = freq.columns.astype(str).str.replace('0', 'total_count')
        freq['%total'] = cls.give_percentage(freq['total_count'])

        fig, (l, r) = plt.subplots(1, 2, figsize=(15, 4), dpi=150)
        cls.plot_column(freq[x], freq['total_count'], condition_on=freq[y],
                    annotate=False, axis=l,  paletter={0: 'black', 1: 'gray'},
                    plot_title='Relationship Between Month and Autumn')
        l.set_ylim(top=32000)
        sns.move_legend(l, (0.05, 0.8))
        
        #8. is_dst and season
        x, y = ['is_dst', 'season']

        freq = df[[x, y]].value_counts().sort_index().reset_index()
        freq.columns = freq.columns.astype(str).str.replace('0', 'total_count')
        freq['%total'] = cls.give_percentage(freq['total_count'])

        fig, (l, r) = plt.subplots(1, 2, figsize=(25, 14), dpi=150)
        cls.plot_column(freq[x], freq['total_count'], condition_on=freq[y],
                    axis=l,  paletter={0: 'black', 1: 'gray'}, annotate=False,
                    plot_title='Relationship Between DST and Autumn')
        line1 = 'Autumn accidents were HIGHER during DST than before and after DST.\n'  
        line2 = 'Hence, a positive relationship.'
        annot = line1+line2
        l.text(-0.47, 138000, annot, color='blue', size=14, weight='bold',
                 bbox={'edgecolor':'black', 'facecolor':'none'})
        l.set_ylim(top=150000)

        
        
    @staticmethod
    def visualize_distributions(df, condition_on=None, savefig=False):
        """display distributions of each numeric column by plotting a
        scatterplot for columns > 15 categories, and columnplot for
        columns <=15 categories."""

        cls = RoadAccidents()

        for col in list(df.select_dtypes(include='number').columns):
            freq = df[col].value_counts().round(2).sort_index()
            if len(freq.index) > 30:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), dpi=150)
                cls.plot_scatter(freq.index, freq, condition_on=condition_on, plot_title=f'Distribution of {col.capitalize()}', 
                                  x_labe=col, y_labe='Count', axis=ax1)
                cls.plot_box(df[col], plot_title=f'{col.capitalize()} Boxplot', axis=ax2)
                plt.show()
            elif 20 < len(freq.index) <= 30:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), dpi=150)
                cls.plot_hist(df[col], condition_on=condition_on, plot_title=f'Distribution of {col.capitalize()}', 
                                  x_labe=col, y_labe='Count', bins=10, axis=ax1)
                cls.plot_box(df[col], plot_title=f'{col.capitalize()} Boxplot', axis=ax2)
                plt.show()
            else:
                fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=150)
                cls.plot_column(freq.index, freq, plot_title=f'Distribution of {col.capitalize()}', 
                               x_labe=col, y_labe='Count', #top_labe_gap=10,
                                  axis=ax)
                plt.show()
            if savefig:
                new_words, words = list(), col.split('_')
                for word in words:
                    chars = "".join([char for char in word if char not in string.punctuation])
                    new_words.append(chars)
                fname = str.lower('_'.join(new_words))
                fig_filename = f"distr_{fname}.png"
                print(cls.fig_writer(fig_filename, fig))
            
    @staticmethod
    def generate_aggregated_lookup(df, using_cols: list=None):
        """generate a lookup table using a subset of variables
        comprising a total accident count per category.
        Return:
        freq_lookup_df"""
        
        if not using_cols:
            using_cols = ['month', 'quarter', 'is_dst', 'is_autumn', 'is_spring', 'is_winter', 'is_summer',
                          'week_num', 'day_of_week', 'day_name', 'is_weekend', 'hour', 'day_num']
        
        aggregate_df = df[using_cols].value_counts().sort_index().reset_index()
        aggregate_df.columns = aggregate_df.columns.astype(str).str.replace('0', 'total_count')
        return aggregate_df
        
    @staticmethod
    def engineer_useful_features(acc_df, lookup_df):
        """engineer the following features from existing columns
        1. hour and minute from the time column 
        2. full_hour from hour and minute columns
        3. month, day, month_name, week_num, day_num and day_name
        from the date column."""
        
        cls = RoadAccidents()
        df = pd.DataFrame(acc_df)
        #1. derive hour, minute from time column
        cols = ['hour', 'minute']
        df[cols] = cls.split_time_series(df['time']).iloc[:, :-1].astype(int)
        df[cols]
        
        #2. combine hour and minute to form full_hour column
        cols = 'full_hour'
        df[cols] = (df.apply(lambda row: row['hour'] + row['minute']/60,
                                         axis=1)).round(4)
        df[cols]
        
        #3. derive day, month from date column
        cols = ['month', 'day']
        df[cols] = cls.split_date_series(df.date.astype(str), sep='-', year_first=True).iloc[:, 1:].astype(int)
        df[cols]
        
        # 4. derive quarter from month_num
        # quarterly accident report
        guide = {1:1, 2:1, 3:1, 4:2, 5:2, 6:2, 
                 7:3, 8:3, 9:3, 10:4, 11:4, 12:4}
        cols = 'quarter'
        df[cols] = df['month'].apply(lambda x: guide[x])
        
        #5. derive month_name from date column
        cols = 'month_name'
        df[cols] = df['date'].astype(np.datetime64).dt.month_name()
        df[cols]
        
        #6. derive week_number from date 
        df['week_num'] = df['date'].astype(np.datetime64).dt.isocalendar().week
#         print(sorted(df.week_num.unique()))
        
        #7. derive days of the year from date
        df['day_num'] = df['date'].astype(np.datetime64).dt.day_of_year
        df.day_num
        
        #8. derive day_name from date using Day of Week in the lookup spreadsheet
        # as dictionary guide
        guide = dict(lookup_df['Day of Week'].to_records(index=False))
        df['day_name'] = df.apply(lambda row: guide[row['day_of_week']], axis=1)
        
        #9. derive is_weekend from day_name
        guide = {'Sunday': 1, 'Monday': 0, 'Tuesday': 0, 'Wednesday': 0, 'Thursday': 0, 'Friday': 0, 'Saturday': 1}
        df['is_weekend'] = df['day_name'].apply(lambda x: guide[x])
        
        #10. derive seasons from month and day
        df['season'] = cls.assign_seasons(df[['month', 'day']])
        
        #11. derive DST indicator variable
        df['is_dst'] = cls.select_dst_period(df[['month_name', 'day', 'hour']]).astype(int)
        
        return df.drop(['time', 'date'],
                      axis=1)
        
    @staticmethod
    def plot_hist(x=None, y=None, condition_on=None, plot_title="A Histogram", bins=None, interval_per_bin: int=None, 
                  bin_range=None, color=None, layer_type='default', x_labe=None, y_labe=None, axis=None, figsize=(8, 4), dpi=150):
        """A histogram plot on an axis.
        layer_type: {"layer", "dodge", "stack", "fill"}
        Return 
        axis"""
        
        if (not isinstance(x, (pd.Series, np.ndarray)) and x is not None):
            raise TypeError("x must be a pandas series or numpy array")
        elif (not isinstance(y, (pd.Series, np.ndarray)) and y is not None):
            raise TypeError("y must be a pandas series or numpy array")
            
        if not bins:
            bins = 'auto'
        if str.lower(layer_type) == 'default':
            layer_type = 'layer'
        if not axis:
            plt.figure(figsize=figsize, dpi=dpi)
            axis = sns.histplot(x=x, y=y, hue=condition_on, bins=bins, binrange=bin_range, 
                                binwidth=interval_per_bin, color=color, multiple=layer_type)
        else:
            sns.histplot(x=x, y=y, hue=condition_on, bins=bins, binrange=bin_range,
                         binwidth=interval_per_bin, color=color, multiple=layer_type, 
                         ax=axis)
        axis.set_title(plot_title, x=0.5, y=1.025)
        if x_labe:
            axis.set_xlabel(str.capitalize(x_labe), weight='bold')
        if y_labe:
            axis.set_ylabel(str.capitalize(y_labe), weight='bold')
            
        return axis
        
    @staticmethod
    def correlation_analyser(df):
        """analyse relationship among variables in three steps.
        1. visualize relationship among all variables
        2. visualize relationship of variables with correlation >=50%
        3. visualize relationship of variables with correlation >=85%,
        indicating autocorrelation"""
        
        cls = RoadAccidents()
        #1. Overall correlation
        fig = plt.figure(figsize=(80, 40), dpi=200)
        cls.plot_correl_heatmap(df, 'OVERVIEW OF THE RELATIONSHIPS AMONG ALL VARIABLES', 
                             annot_size=20, xticklabe_size=28, yticklabe_size=28, title_size=38)
        plt.show()
        print(cls.fig_writer('overall_correlations.png', fig))
        
        #2. select variables with 50% and above correlation
        correlated_vars = cls.select_corr_with_threshold(df, 0.5)
        print(correlated_vars.columns)
        
        fig = plt.figure(figsize=(40, 20), dpi=200)
        cls.plot_correl_heatmap(correlated_vars, 'Correlated Features in the accident dataframe',
                           xticklabe_size=16, yticklabe_size=16, annot_size=20, title_size=28, 
                                run_correlation=False)
        plt.show()
        print(cls.fig_writer('correlated_features.png', fig))
        
        #3. select variables with 85% and above correlation
        # indicating autocorrelation
        auto_corr = cls.select_corr_with_threshold(df, 0.85)
        print(auto_corr.columns)
        fig = plt.figure(figsize=(20, 10), dpi=200)
        cls.plot_correl_heatmap(auto_corr, 'Auto Correlated Features in the accident dataframe',
                           xticklabe_size=16, yticklabe_size=16, annot_size=16, title_size=20,
                                run_correlation=False)
        plt.show()
        print(cls.fig_writer('auto_correlated_features.png', fig))
        
    @staticmethod
    def select_corr_with_threshold(df: pd.DataFrame, thresh: float=0.5, report_only_colnames=False):
        """select only variables with correlation equal to or above the threshold value.
        Return:
        df_corr"""
        
        def count_thresh_corrs(row, thresh):
            """to count how many values in each row >= thresh"""
            
            count = 0
            for val in row:
                if abs(val) >= abs(thresh):
                    count += 1
            return count
        
        df = pd.DataFrame(df)
        if not isinstance(df, pd.DataFrame):
            raise TypeError('df must be a dataframe')
            
        print(f'Features with correlation of {round(100*thresh, 2)}% and above:\n')
        # number of values >= threshold value per row
        all_corrs = df.corr().round(4)
        selected_corr = all_corrs.apply(lambda val: count_thresh_corrs(val, thresh))
        correlated_vars = selected_corr[selected_corr > 1]
        if report_only_colnames:
            return list(correlated_vars.index)
        return all_corrs.loc[all_corrs.index.isin(list(correlated_vars.index)), list(correlated_vars.index)]
        
    @staticmethod
    def char_cleanup(col_name: str):
        
        if '_' not in col_name:
            return col_name
            
        new_words, words = list(), col_name.split('_')
        for word in words:
            chars = "".join([char for char in word if char not in string.punctuation])
            new_words.append(chars)
        return str.lower('_'.join(new_words))
        
    @staticmethod
    def cleanup_cols(df):
        """remove all unwanted characters from column names in lowercase
        Return:
        column_names: cleaned column names."""
        
        cls = RoadAccidents()
        col_names = list(df.columns.astype(str).str.replace('-', '_'))
        cols_lower = list(map(str.lower, col_names))
        return list(map(cls.char_cleanup, cols_lower))
        
    @staticmethod
    def systematically_impute_all_nans(nan_df):
        """impute missing values in a feature of nan_df with its mean (for continuous) or
        modal value (for categorical) per unique category of the top 3 most correlated
        features."""
        
        df = copy.deepcopy(nan_df)
        cls = RoadAccidents()
        print(f"\n\n{list(cls.null_checker(df, only_nulls=True).index)}")
        # 1. 1st_point_of_impact
        na_cols = '1st_point_of_impact'
        print(f"\n\nIMPUTING {na_cols.upper()} WITH THE MODAL VALUE PER COMBINED CATEGORY OF:")
        # replace 1st_point_of_impact null values with its modal values
        # per unique combination of top 3 most correlated variables
        df = cls.impute_missing_variables(df, na_cols, is_categorical=True)
        print(f'\n{na_cols.upper()} CLEANED!')


        # 2. Junction Location
        print(f"\n\n{list(cls.null_checker(df, only_nulls=True).index)}")
        na_cols = 'junction_location'
        print(f"\n\nIMPUTING {na_cols.upper()} WITH THE MODAL VALUE PER COMBINED CATEGORY OF:")
        # replace junction_location null values with its modal values
        # per unique combination of top 3 most correlated variables
        df = cls.impute_missing_variables(df, na_cols, is_categorical=True)
        print(f'\n{na_cols.upper()} CLEANED!')


        # 3. Hit Object Off Carriageway
        print(f"\n\n{list(cls.null_checker(df, only_nulls=True).index)}")
        na_cols = 'hit_object_off_carriageway'
        print(f"\n\nIMPUTING {na_cols.upper()} WITH THE MODAL VALUE PER COMBINED CATEGORY OF:")
        # replace hit_object_off_carriageway null values with its modal values
        # per unique combination of top 3 most correlated variables
        df = cls.impute_missing_variables(df, na_cols, is_categorical=True)
        print(f'\n{na_cols.upper()} CLEANED!')


        # 4. Vehicle Location Restricted Lane
        print(f"\n\n{list(cls.null_checker(df, only_nulls=True).index)}")
        na_cols = 'vehicle_location_restricted_lane'
        print(f"\n\nIMPUTING {na_cols.upper()} WITH THE MODAL VALUE PER COMBINED CATEGORY OF:")
        # replace vehicle_location-restricted_lane null values with its modal values
        # per unique combination of top 3 most correlated variables
        df = cls.impute_missing_variables(df, na_cols, is_categorical=True)
        print(f'\n{na_cols.upper()} CLEANED!')


        # 5. Hit Object in Carriageway
        print(f"\n\n{list(cls.null_checker(df, only_nulls=True).index)}")
        na_cols = 'hit_object_in_carriageway'
        print(f"\n\nIMPUTING {na_cols.upper()} WITH THE MODAL VALUE PER COMBINED CATEGORY OF:")
        # replace hit_object_in_carriageway null values with its modal values
        # per unique combination of top 3 most correlated variables
        df = cls.impute_missing_variables(df, na_cols, is_categorical=True)
        print(f'\n{na_cols.upper()} CLEANED!')


        # 6. Vehicle Leaving Carriageway
        print(f"\n\n{list(cls.null_checker(df, only_nulls=True).index)}")
        na_cols = 'vehicle_leaving_carriageway'
        print(f"\n\nIMPUTING {na_cols.upper()} WITH THE MODAL VALUE PER COMBINED CATEGORY OF:")
        # replace vehicle_leaving_carriageway null values with its modal values
        # per unique combination of top 3 most correlated variables
        df = cls.impute_missing_variables(df, na_cols, is_categorical=True)
        print(f'\n{na_cols.upper()} CLEANED!')


        # 7. Vehicle Manoeuvre
        print(f"\n\n{list(cls.null_checker(df, only_nulls=True).index)}")
        na_cols = 'vehicle_manoeuvre'
        print(f"\n\nIMPUTING {na_cols.upper()} WITH THE MODAL VALUE PER COMBINED CATEGORY OF:")
        # replace vehicle_manoeuvre null values with its modal values
        # per unique combination of top 3 most correlated variables
        df = cls.impute_missing_variables(df, na_cols, is_categorical=True)
        print(f'\n{na_cols.upper()} CLEANED!')


        # 8. Skidding and Overturning
        print(f"\n\n{list(cls.null_checker(df, only_nulls=True).index)}")
        na_cols = 'skidding_and_overturning'
        print(f"\n\nIMPUTING {na_cols.upper()} WITH THE MODAL VALUE PER COMBINED CATEGORY OF:")
        # replace skidding_and_overturning null values with its modal values
        # per unique combination of top 3 most correlated variables
        df = cls.impute_missing_variables(df, na_cols, is_categorical=True)
        print(f'\n{na_cols.upper()} CLEANED!')


        # 9. Was Vehicle Left Hand Drive
        print(f"\n\n{list(cls.null_checker(df, only_nulls=True).index)}")
        na_cols = 'was_vehicle_left_hand_drive'
        print(f"\n\nIMPUTING {na_cols.upper()} WITH THE MODAL VALUE PER COMBINED CATEGORY OF:")
        # replace was_vehicle_left_hand_drive? null values with its modal values
        # per unique combination of top 3 most correlated variables
        df = cls.impute_missing_variables(df, na_cols, is_categorical=True)
        print(f'\n{na_cols.upper()} CLEANED!')


        # 10. LSOA of Accident Location
        print(f"\n\n{list(cls.null_checker(df, only_nulls=True).index)}")
        na_cols = 'lsoa_of_accident_location'
        print(f"\n\nIMPUTING {na_cols.upper()} WITH THE MODAL VALUE PER COMBINED CATEGORY OF:")
        # replace lsoa_of_accident_location null values with its modal values
        # per unique combination of pivot columns
        pivot_cols = ['local_authority_district', 'local_authority_highway']
        df[na_cols] = cls.impute_null_values(df, pivot_cols, na_cols, stat_used='mode')
        print(f'\n{na_cols.upper()} CLEANED!')


        # 11. Casualty IMD Decile
        print(f"\n\n{list(cls.null_checker(df, only_nulls=True).index)}")
        na_cols = 'casualty_imd_decile'
        print(f"\n\nIMPUTING {na_cols.upper()} WITH THE MODAL VALUE PER COMBINED CATEGORY OF:")
        # replace casualty_imd_decile null values with its modal values
        # per unique combination of top 3 most correlated variables
        df = cls.impute_missing_variables(df, na_cols, is_categorical=True)
        print(f'\n{na_cols.upper()} CLEANED!')


        # 12. Age of Driver
        print(f"\n\n{list(cls.null_checker(df, only_nulls=True).index)}")
        na_cols = 'age_of_driver'
        print(f"\n\nIMPUTING {na_cols.upper()} WITH THE MEDIAN VALUE PER COMBINED CATEGORY OF:")
        # replace age_of_driver null values with its median values
        # per unique combination of top 3 most correlated variables
        df = cls.impute_missing_variables(df, na_cols, is_categorical=False, use_mean=False)
        print(f'\n{na_cols.upper()} CLEANED!')


        # 13. Age Band of Driver
        print(f"\n\n{list(cls.null_checker(df, only_nulls=True).index)}")
        na_cols = 'age_band_of_driver'
        print(f"\n\nIMPUTING {na_cols.upper()} WITH THE MODAL VALUE PER COMBINED CATEGORY OF:")
        # replace age_band_of_driver null values with its modal values
        # per unique combination of top 3 most correlated variables
        df = cls.impute_missing_variables(df, na_cols, is_categorical=True)
        print(f'\n{na_cols.upper()} CLEANED!')


        # 14. Casualty Home Area Type
        print(f"\n\n{list(cls.null_checker(df, only_nulls=True).index)}")
        na_cols = 'casualty_home_area_type'
        print(f"\n\nIMPUTING {na_cols.upper()} WITH THE MODAL VALUE PER COMBINED CATEGORY OF:")
        # replace casualty_home_area_type null values with its modal values
        # per unique combination of top 3 most correlated variables
        df = cls.impute_missing_variables(df, na_cols, is_categorical=True)
        print(f'\n{na_cols.upper()} CLEANED!')


        # 15. Driver IMD Decile
        print(f"\n\n{list(cls.null_checker(df, only_nulls=True).index)}")
        na_cols = 'driver_imd_decile'
        print(f"\n\nIMPUTING {na_cols.upper()} WITH THE MODAL VALUE PER COMBINED CATEGORY OF:")
        # replace driver_imd_decile null values with its modal values
        # per unique combination of top 3 most correlated variables
        df = cls.impute_missing_variables(df, na_cols, is_categorical=True)
        print(f'\n{na_cols.upper()} CLEANED!')


        # 16. Vehicle IMD Decile
        print(f"\n\n{list(cls.null_checker(df, only_nulls=True).index)}")
        na_cols = 'vehicle_imd_decile'
        print(f"\n\nIMPUTING {na_cols.upper()} WITH THE MODAL VALUE PER COMBINED CATEGORY OF:")
        # replace vehicle_imd_decile null values with its modal values
        # per unique combination of top 3 most correlated variables
        df = cls.impute_missing_variables(df, na_cols, is_categorical=True)
        print(f'\n{na_cols.upper()} CLEANED!')


        # 17. Driver Home Area Type
        print(f"\n\n{list(cls.null_checker(df, only_nulls=True).index)}")
        na_cols = 'driver_home_area_type'
        print(f"\n\nIMPUTING {na_cols.upper()} WITH THE MODAL VALUE PER COMBINED CATEGORY OF:")
        # replace driver_home_area_type null values with its modal values
        # per unique combination of top 3 most correlated variables
        df = cls.impute_missing_variables(df, na_cols, is_categorical=True)
        print(f'\n{na_cols.upper()} CLEANED!')


        # 18. Propulsion Code
        print(f"\n\n{list(cls.null_checker(df, only_nulls=True).index)}")
        na_cols = 'propulsion_code'
        print(f"\n\nIMPUTING {na_cols.upper()} WITH 0 (PLACEHOLDER FOR UNDEFINED):")
        # replace propulsion_code null values with 0
        # as placeholder for Undefined
        df.loc[df[na_cols].isnull(), na_cols] = 0
        print(f'\n{na_cols.upper()} CLEANED!')


        # 19. Engine Capacity (cc)
        print(f"\n\n{list(cls.null_checker(df, only_nulls=True).index)}")
        na_cols = 'engine_capacity_cc'
        print(f"\n\nIMPUTING {na_cols.upper()} WITH THE MODAL VALUE PER COMBINED CATEGORY OF:")
        # replace engine_capacity_ null values with its modal values
        # per unique combination of top 3 most correlated variables
        df = cls.impute_missing_variables(df, na_cols, is_categorical=True)
        print(f'\n{na_cols.upper()} CLEANED!')

        # 20. Age of Vehicle
        print(f"\n\n{list(cls.null_checker(df, only_nulls=True).index)}")
        na_cols = 'age_of_vehicle'
        print(f"\n\nIMPUTING {na_cols.upper()} WITH THE MEDIAN VALUE PER COMBINED CATEGORY OF:")
        # replace age_of_vehicle null values with its median values
        # per unique combination of top 3 most correlated variables
        df = cls.impute_missing_variables(df, na_cols, is_categorical=False, use_mean=False)
        print(f'\n{na_cols.upper()} CLEANED!')


        # 21. 2nd Road Class
        print(f"\n\n{list(cls.null_checker(df, only_nulls=True).index)}")
        na_cols = '2nd_road_class'
        print(f"\n\nIMPUTING {na_cols.upper()} WITH 6 CODE FOR UNCLASSIFIED CATEGORY:")
        # replace 2nd_road_class null values with 6
        # representing the Unclassified category
        df.loc[df[na_cols].isnull(), na_cols] = 6
        print(f'\n{na_cols.upper()} CLEANED!')


        # 22. Junction Control
        print(f"\n\n{list(cls.null_checker(df, only_nulls=True).index)}")
        na_cols = 'junction_control'
        print(f"\n\nIMPUTING {na_cols.upper()} WITH THE MODAL VALUE PER COMBINED CATEGORY OF:")
        # replace junction_control null values with its modal values
        # per unique combination of top 3 most correlated variables
        df = cls.impute_missing_variables(df, na_cols, is_categorical=True)
        print(f'\n{na_cols.upper()} CLEANED!')
        print('\n\nDATA CLEANING COMPLETE!')
        
        return df.reset_index(drop=True)
        
    @staticmethod
    def placeholder_to_nan(df, current_nan_char: str=-1, new_placeholder=np.nan):
        """convert placeholders values to np.nan
        Return:
        df: dataframe with placeholder replaced by -1."""
        
        col_names = df.columns
        df2 = copy.deepcopy(df)
        for col in col_names:
    #         print(col)
            df2.loc[df2[col] == current_nan_char, col] = new_placeholder
        return df2
        
    @staticmethod
    def impute_missing_variables(df, col_with_nan:str, is_categorical:bool=True, use_mean: bool=True):
        """impute missing entries of variable with its mean or modal values
        conditioned on the top 3 most correlated features."""
        
        cls = RoadAccidents()
        na_cols = col_with_nan
        
        cls.get_casualties_for_nan_col(df, na_cols)
        
        top3_corr = cls.corr_with_kbest(df.dropna().select_dtypes(include='number').drop(na_cols, axis=1),
                                     df.dropna()[na_cols]).iloc[:3].index
        print(f"Top 3 correlated variables used to calculate modes:\n{list(top3_corr)}")
        
        # replace junction_location null values with its modal values
        # per unique combination of top 3 most correlated variables
        corr_cols = list(top3_corr)
        if is_categorical:
            stat_used = 'mode'
        else:
            if use_mean:
                stat_used = 'mean'
            else:
                stat_used = 'median'
        df[na_cols] = cls.impute_null_values(df, corr_cols, na_cols, stat_used=stat_used)
        return df
        
    @staticmethod
    def get_casualties_for_nan_col(df, null_colname:list):
        """get all casualties having same accident index."""
        
        acc_ix = df.loc[df[null_colname].isnull(), 'accident_index'].values
    #     print(acc_ix)
        return df.loc[df['accident_index'].isin(acc_ix), ['accident_index', null_colname]]
        
    @staticmethod
    def impute_null_values(df: 'pd.DataFrame', pivot_cols: list, target_col: str, stat_used: str='mean'):
        """Impute the null values in a target feature using aggregated values
        (eg mean, median, mode) based on pivot features
        :param df:
        :param pivot_cols:
        :param target_col:
        :param with_stat: {'mean', 'mode', 'median'}
        :return: impute_guide: 'a pandas dataframe'"""
        
        cls = RoadAccidents()
        
        if ('object' in str(df[target_col].dtypes)) or ('bool' in str(df[target_col].dtypes)):
            stat_used = 'mode'

        if str.lower(stat_used) == 'mean':
            impute_guide = cls.pivot_mean(df, pivot_cols, target_col)
        elif str.lower(stat_used) == 'mode':
            impute_guide = cls.pivot_mode(df, pivot_cols, target_col)
        elif str.lower(stat_used) == 'median':
            impute_guide = cls.pivot_median(df, pivot_cols, target_col)
        # fill null values with means
        null_ind = df.loc[df[target_col].isnull()].index

        piv_rec = pd.merge(left=df.loc[null_ind, pivot_cols],
                           right=impute_guide,
                           how='left', on=pivot_cols)
        # fill all lingering null values with general mean
        if piv_rec[target_col].isnull().sum():
            piv_rec.loc[piv_rec[target_col].isnull(), target_col] = impute_guide[target_col].mode().values[0]

        piv_rec.index = null_ind

        return pd.concat([df.loc[df[target_col].notnull(), target_col],
                          piv_rec[target_col]],
                         join='inner', axis=0).sort_index()
    
    @staticmethod
    def pivot_mode(df: 'pd.DataFrame', pivot_cols: list, target_col: str):
        """rank the occurrences of target_col values based on pivot_cols,
        and return the mode (i.e the highest occurring) target_col
        value per combination of pivot_cols.
        Return:
        impute_guide: lookup table"""
        
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Object given is not a pandas dataframe")
        if not isinstance(pivot_cols, list):
            raise TypeError("pivot columns should be in list")
        if not isinstance(target_col, str):
            raise TypeError("Target column name must be a string")

        freq_df = df.loc[df[target_col].notnull(), pivot_cols + [target_col]].value_counts().reset_index()
        return freq_df.drop_duplicates(subset=pivot_cols).drop(labels=[0], axis=1)
        
    @staticmethod
    def pivot_mean(df: 'pd.DataFrame', pivot_cols: list, target_col: str):
        """rank the occurrences of target_col values based on pivot_cols,
        and return the average target_col value per combination
        of pivot_cols.
        Return:
        impute_guide: lookup table"""
        
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Object given is not a pandas dataframe")
        if not isinstance(pivot_cols, list):
            raise TypeError("pivot columns should be in list")
        if not isinstance(target_col, str):
            raise TypeError("Target column name must be a string")
        
        dec_places = 4
        targ_dtype = str(df[target_col].dtypes)
        
        if ('int' in targ_dtype):
            dec_places = 0
            targ_dtype = str(df[target_col].dtypes)[:-2]
            
        elif ('float' in targ_dtype):    
            sample = str(df.loc[df[target_col].notnull(), target_col].iloc[0])
            dec_places = len(sample.split('.')[-1])
            targ_dtype = str(df[target_col].dtypes)[:-2]
        
        freq_df = np.round(df.loc[df[target_col].notnull(), pivot_cols + [target_col]].groupby(by=pivot_cols).mean().reset_index(), dec_places)
        return freq_df.drop_duplicates(subset=pivot_cols)
        
    @staticmethod
    def pivot_median(df: 'pd.DataFrame', pivot_cols: list, target_col: str):
        """rank the occurrences of target_col values based on pivot_cols,
        and return the median target_col value per combination
        of pivot_cols.
        Return:
        impute_guide: lookup table"""
        
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Object given is not a pandas dataframe")
        if not isinstance(pivot_cols, list):
            raise TypeError("pivot columns should be in list")
        if not isinstance(target_col, str):
            raise TypeError("Target column name must be a string")
        
        dec_places = 4
        targ_dtype = str(df[target_col].dtypes)
        
        if ('int' in targ_dtype):
            dec_places = 0
            targ_dtype = str(df[target_col].dtypes)[:-2]
            
        elif ('float' in targ_dtype):    
            sample = str(df.loc[df[target_col].notnull(), target_col].iloc[0])
            dec_places = len(sample.split('.')[-1])
            targ_dtype = str(df[target_col].dtypes)[:-2]
        
        freq_df = np.round(df.loc[df[target_col].notnull(), pivot_cols + [target_col]].groupby(by=pivot_cols).median().reset_index(), dec_places)
        return freq_df.drop_duplicates(subset=pivot_cols)
        
    @staticmethod
    def give_percentage(arr: pd.Series):
        """output the percentage of each element in array"""
        
        ser = pd.Series(arr)
        return np.round(100*ser/ser.sum(), 2)
    
    @staticmethod
    def select_dst_period(df):
        """select records surrounding the 2019 Daylight Savings Time (DST) period.
        2019 DST: 1am March 31 to 2am October 27.
        Rules used:
        1. select all records in April, May, June, July, August, September.
        2. include all records on 31st March except 0am.
        3. include all records from October 1st to 27th.
        4. exclude records beyond 2am of October 27th.
        Return: df_records_in_dst"""
        
        full_months = ('April', 'May', 'June', 'July', 'August', 'September')
        # select everyday in full months
        select_full_months = df['month_name'].isin(full_months)

        # include march 31, 1am - 23pm
        include_march_31_except_0am = ((df['month_name'] == 'March') & 
                                       (df['day'] == 31) &
                                       (df['day'] == 31) &
                                       (df['hour'] != 0))
        # include october, 1st - 27th
        include_october_except_beyond_27 = ((df['month_name'] == 'October') & 
                                            (df['day'] <= 27))
        # exclude 3am and beyond on october 27
        exclude_october_27_3am_and_beyond = ~((df['month_name'] == 'October') & 
                                              (df['day'] == 27) & 
                                              (df['hour'] >= 3))

        intermediate_view = df.loc[select_full_months |
                             include_march_31_except_0am |
                             include_october_except_beyond_27]

        dst_ind = intermediate_view.loc[exclude_october_27_3am_and_beyond].index
        result = pd.Series(index=df.index, dtype=float)
        result.loc[dst_ind] = 1
        result.loc[~result.index.isin(dst_ind)] = 0
        return result
        
    @staticmethod
    def dst_prior_week(df):
        """select records surrounding the 2019 Daylight Savings Time (DST) period.
        2019 DST: 1am March 31 to 2am October 27.
        1 Week (7 days) before onset of DST: 1am 24-March-2019 to 0am 31-March-2019
        Rules used:
        1. select all records on 24, 25, 26, 27, 28, 29, 30
        2. include only records of 0am on 31st March.
        3. exclude records of 0am on 24th March.
        Return: 
        df_week_before_dst"""
        
        full_days = (24, 25, 26, 27, 28, 29, 30)
        # select everyday in full days in March
        select_full_days = ((df['month_name'] == 'March') &
                            (df['day'].isin(full_days)))

        # include march 31, 0am
        include_march_31_0am = ((df['month_name'] == 'March') & 
                                (df['day'] == 31) &
                                (df['hour'] == 0))

        # exclude 0am on March 24
        exclude_march_24_0am = ~((df['month_name'] == 'March') & 
                                 (df['day'] == 24) & 
                                 (df['hour'] == 0))

        intermediate_view = df.loc[select_full_days | include_march_31_0am]

        return intermediate_view.loc[exclude_march_24_0am]
        
    @staticmethod
    def dst_first_week(df):
        """select records surrounding the 2019 Daylight Savings Time (DST) period.
        2019 DST: 1am March 31 to 2am October 27.
        First Week (7 days) of DST: 1am March 31 to 0am April 7
        Rules used:
        1. select all records on April 1, 2, 3, 4, 5, 6
        2. select all records on March 31 beyond 0am
        3. include April 7 0am
        Return: 
        df_first_week"""
        
        march_full_days = 31
        april_full_days = (1, 2, 3, 4, 5, 6)
        
        # select full days in March
        select_full_days_mar = ((df['month_name'] == 'March') &
                                (df['day'] == march_full_days))
        # select full days in April
        select_full_days_apr = ((df['month_name'] == 'April') &
                            (df['day'].isin(april_full_days)))

        # include April 7 0am
        include_apr_7_0am = ((df['month_name'] == 'April') & 
                                         (df['day'] == 7) &
                                         (df['hour'] == 0))

        final_view = df.loc[select_full_days_mar | 
                                   select_full_days_apr |
                                   include_apr_7_0am]

        return final_view
        
    def dst_last_week(df):
        """select records surrounding the 2019 Daylight Savings Time (DST) period.
        2019 DST: 1am March 31 to 2am October 27.
        Last Week (7 days) of DST: October 20 beyond 2am to October 27 before 3am
        Rules used:
        1. select all records on October 21, 22, 23, 24, 25, 26
        2. include records on October 20 beyond 2am
        3. include records on October 27 before 3am
        Return: 
        df_last_week"""
        
        full_days = (21, 22, 23, 24, 25, 26)
        
        # select full days
        select_full_days = ((df['month_name'] == 'October') &
                            (df['day'].isin(full_days)))

        # include October 20 beyond 2am
        include_oct_20_beyond_2am = ((df['month_name'] == 'October') & 
                                         (df['day'] == 20) &
                                         ~(df['hour'].isin([0, 1, 2])))
        # include October 27 before 3am
        include_oct_27_before_3am = ((df['month_name'] == 'October') & 
                                         (df['day'] == 27) &
                                         (df['hour'].isin([0, 1, 2])))

        final_view = df.loc[select_full_days |
                            include_oct_20_beyond_2am |
                            include_oct_27_before_3am]

        return final_view
        
    @staticmethod
    def week_after_dst(df):
        """select records surrounding the 2019 Daylight Savings Time (DST) period.
        2019 DST: 1am March 31 to 2am October 27.
        1 Week (7 days) after end of DST: 3am 27-10-2019 to 2am 3-11-2019
        Rules used:
        1. select all records on October 28, 29, 30, 31
        2. select all records on November 1, 2
        3. include October 27th beyond 2am
        4. include November 3rd before 3am
        Return: 
        df_week_after_dst"""
        
        october_full_days = (28, 29, 30, 31)
        november_full_days = (1, 2)
        # select full days in October
        select_full_days_oct = ((df['month_name'] == 'October') &
                                (df['day'].isin(october_full_days)))
        # select full days in November
        select_full_days_nov = ((df['month_name'] == 'November') &
                            (df['day'].isin(november_full_days)))

        # include October 27, 3am and beyond
        include_october_27_beyond_2am = ((df['month_name'] == 'October') & 
                                         (df['day'] == 27) &
                                         ~(df['hour'].isin([0, 1, 2])))

        # include November 3, 0am to 2am
        include_november_3_before_3am = ((df['month_name'] == 'November') & 
                                          (df['day'] == 3) & 
                                          (df['hour'].isin([0, 1, 2])))

        final_view = df.loc[select_full_days_oct | 
                                   select_full_days_nov |
                                   include_october_27_beyond_2am |
                                   include_november_3_before_3am]

        return final_view
        
    @staticmethod
    def corr_with_kbest(X, y):
        """using sklearn.preprocessing.SelectKBest, quantify correlations
        between features in X and y.
        Return: correlation series"""
        
        X = pd.DataFrame(X)
        selector = s_fs.SelectKBest(k='all').fit(X, y)
        return pd.Series(selector.scores_, index=selector.feature_names_in_).sort_values(ascending=False)
        
    @staticmethod
    def cast_categs_to_int(ser: pd.Series):
        """cast categories to integers
        ser: categorical series
        Return: integer-encoded series"""
        
        int_encoder = s_prep.LabelEncoder()
        return pd.Series(int_encoder.fit_transform(ser))
        
    @staticmethod
    def get_features_with_dtypes(df, feat_datatype: str='number'):
        """get a list of features with specified datatype in the dataframe.
        default output is numeric features.
        feat_datatype: options are number/int/float, object/str, bool, datetime/date/time
        Return: feat_list"""
        
        if not isinstance(df, pd.DataFrame):
            raise TypeError('df must be a dataframe')
            
        if not isinstance(feat_datatype, str):
            feat_datatype = eval(str(feat_datatype).split()[1][:-1])
        
        guide = {'str': 'object', 'int': 'number', 'float': 'number', 'date': 'datetime', 'time': 'datetime'}
        if feat_datatype in guide:
            use_datatype = guide[feat_datatype]
            print(use_datatype)
            cols_selected = df.select_dtypes(include=use_datatype)
        else:
            cols_selected = df.select_dtypes(include=feat_datatype)
        
        
        if feat_datatype in ['int', 'float']:
            
            #print(feat_datatype)
            col_dtypes = cols_selected.dtypes.astype(str).str.lower()
            return list(col_dtypes[col_dtypes.str.contains(feat_datatype)].index)
        #print('yes')
        return list(cols_selected.columns)
        
    @staticmethod
    def report_a_significance(X1_set, X2_set, n_deg_freedom=1, X1_name='X1', X2_name='X2'):
        """Test for statistical significant difference between X1_set and X2_set
        at 99% and 95% Confidence.
        X1_set: 1D array of observations
        X2_set: 1D array of observations."""
        
        cls = DsUtils()
        
        def get_min_denom(n1, n2):
            return min([n1, n2])
        
        def detect_unequal_sizes(n1, n2):
            """check if both lengths are not the same"""
            return n1 != n2
        
        # to ensure reproducibility
        np.random.seed(1)
        
        X1_size, X2_size = len(X1_set), len(X2_set)
        samp_sizes = {X1_name: pd.Series(X1_set), 
                      X2_name: pd.Series(X2_set)}
        print(f'ORIGINAL SAMPLE SIZE: \n{X1_name}: {X1_size}\n' +
              f'{X2_name}: {X2_size}\n\n')
        
        # check if sample sizes are unequal
        if detect_unequal_sizes(X1_size, X2_size):
            print("Unequal Sample Sizes Detected!!")
            min_size = get_min_denom(X1_size, X2_size)
            max_samp_name = [name for name, val in samp_sizes.items() if len(val) != min_size][0]
            # downsampling: 
            # randomly generate min_size indexes for max_samp_name
            rand_indexes = cls.index_generator(len(samp_sizes[max_samp_name]), min_size)
            # select only random min_size indexes for max_samp_name set
            samp_sizes[max_samp_name] = samp_sizes[max_samp_name].iloc[rand_indexes]
            X1_size, X2_size = len(samp_sizes[X1_name]), len(samp_sizes[X2_name])
            print(f'ADJUSTED SAMPLE SIZE: \n{X1_name}: {X1_size}\n' +
                  f'{X2_name}: {X2_size}\n\n')

        total_X1, X1_mean, X1_std = cls.compute_mean_std(samp_sizes[X1_name], X1_size, n_deg_freedom)
        total_X2, X2_mean, X2_std = cls.compute_mean_std(samp_sizes[X2_name], X2_size, n_deg_freedom)
        
        null_hypo = np.round(X1_mean - X2_mean, 4)
        pooled_std = cls.compute_pstd(X1_std, X2_std)

        print(f'{X1_name}:\n Total = {total_X1}\n Average = {X1_mean}\n Standard deviation = {X1_std}\n\n' +
              f'{X2_name}:\n Total = {total_X2}\n Average = {X2_mean}\n Standard deviation = {X2_std}\n\n' +
              f'MEAN DIFFERENCE = {null_hypo}\n' +
              f'POOLED STD = {pooled_std}\n\n')
        
        print(f'HYPOTHESIS TEST:\nIs {X1_mean} significantly HIGHER THAN {X2_mean}?\n' +
             f'BASED ON the chosen level of significance\nIs the difference {null_hypo} > 0?\n')

        # check for both 99% and 95% confidence level
        # Meaning, is the difference between both figures greater than 
        # 3 pooled std and 2 pooled std respectively

        alpha = 0.01
        test_result = cls.compute_test(pooled_std, alpha)
        if null_hypo > test_result[1]:
            return print(f'At {test_result[0]}, REJECT the null hypothesis!\n {null_hypo} is greater than {test_result[1]}\n')
        else:
            test_result = cls.compute_test(pooled_std)
            if null_hypo > test_result[1]:
                return print(f'At {test_result[0]}, REJECT the null hypothesis!\n{null_hypo} is greater than {test_result[1]}\n')
        print(f'Do NOT reject the null hypothesis\n{null_hypo} is less than or equal to {test_result[1]}')
        
    @staticmethod
    def calc_deg_freedom(denom, n_deg):
        """compute degrees of freedom."""
        
        return denom - n_deg
        
    @staticmethod
    def compute_mean_std(arr, denom, n_deg):
        """compute sum, mean, stdev of array using 
        the given denominator and degrees of freedom"""
        
        cls = RoadAccidents()
        
        total = np.sum(arr)
        avg = np.round(total/denom, 4)
        deg_freedom = cls.calc_deg_freedom(denom, n_deg)
        sumsq = np.sum((arr - avg)**2)
        stdv = np.sqrt(sumsq/deg_freedom).round(4)
        return (total, avg, stdv)
        
    @staticmethod
    def compute_pstd(stdv_1, stdv_2):
        """Compute pooled standard devs from two stdevs"""
        
        return round(np.sum([stdv_1**2, stdv_2**2])**0.5, 4)
    
    @staticmethod
    def compute_test(pooled_stdv, at_alpha=0.05):
        """Compute test sigma at specified significance with pooled_std.
        at_alpha: significance
        pooled_std: pooled standard deviation
        Return: (confidence, test_sigma)"""
        
        sig_to_conf = {0.05: (2, '95% confidence'),
                      0.01: (3, '99% confidence')}
        test_sigma = round(sig_to_conf[at_alpha][0] * pooled_stdv, 4)
        return (sig_to_conf[at_alpha][1], test_sigma)
    
    @staticmethod
    def index_generator(sample_size: 'array', n_index=1):
        """Randomly generate n indexes.
        :Return: random_indexes"""

        import random

        def select_from_array(sample_array, n_select=1):
            return random.choices(population=sample_array, k=n_select)

        indexes = range(0, sample_size, 1)

        return select_from_array(indexes, n_index)
    
    @staticmethod
    def split_time_series(time_col: 'datetime series', sep=':'):
        """split series containing time data in str format into
        dataframe of three columns (hour, minute, seconds)
        Return:
        Dataframe"""
        
        time_col = time_col.astype(np.datetime64)
        hr, minute, secs = time_col.dt.hour, time_col.dt.minute, time_col.dt.second
        return pd.DataFrame({'Hour': hr,
                            'Minute': minute,
                            'Seconds': secs})
    
    @staticmethod
    def plot_scatter(x, y, condition_on=None, plot_title='Scatter Plot', marker=None, color=None, 
                     paletter='viridis', x_labe=None, y_labe=None, axis=None, figsize=(8, 4), dpi=150,
                     savefig=False, fig_filename='scatterplot.png'):
        """plot scatter graph on an axis.
        Return: axis """
        
        if not axis:
            plt.figure(figsize=figsize, dpi=dpi)
            axis = sns.scatterplot(x=x, y=y, hue=condition_on, marker=marker, palette=paletter, color=color)
        else:
            sns.scatterplot(x=x, y=y, hue=condition_on, marker=marker,
                                  ax=axis, palette=paletter, color=color)
        axis.set_title(plot_title, weight='bold', x=0.5, y=1.025)
        if x_labe:
            axis.set_xlabel(str.capitalize(x_labe), weight='bold')
        if y_labe:
            axis.set_ylabel(str.capitalize(y_labe), weight='bold')
            
        if savefig:
            print(cls.fig_writer(fig_filename, fig))
        return axis
        
    @staticmethod
    def plot_line(x, y, condition_on=None, plot_title='Line Plot', line_size=None, 
                  legend_labe=None, show_legend=False, marker=None, color=None, 
                  x_labe=None, y_labe=None, axis=None):
        """plot line graph on an axis.
        Return: axis """
        
        if not axis:
            axis = sns.lineplot(x=x, y=y, hue=condition_on, marker=marker, size=line_size, label=legend_labe, 
                               legend=show_legend, palette='viridis', color=color)
        else:
            sns.lineplot(x=x, y=y, hue=condition_on, marker=marker, size=line_size, legend=show_legend,
                                  ax=axis, palette='viridis', color=color, label=legend_labe)
        
        axis.set_title(plot_title, weight='bold')
        if x_labe:
            axis.set_xlabel(str.capitalize(x_labe), weight='bold')
        if y_labe:
            axis.set_ylabel(str.capitalize(y_labe), weight='bold')
        return axis
                          
    @staticmethod
    def plot_column(x, y, condition_on=None, plot_title='A Column Chart', x_labe=None, y_labe=None, annotate=True,
                color=None, paletter='viridis', conf_intvl=None, include_perc=False, perc_freq=None,
                annot_size=6, top_labe_gap=None, bot_labe_gap=None, h_labe_shift=0.4, top_labe_color='black',
                bot_labe_color='blue', index_order: bool=True, rotate_xticklabe=False, axis=None, 
                figsize=(8, 4), dpi=150, savefig=False, fig_filename='columnplot.png'):
        """plot bar graph on an axis.
        If include_perc is True, then perc_freq must be provided.
        :Return: axis """
        
        freq_col = y
        
        if color:
            paletter = None
        if not axis:
            plt.figure(figsize=figsize, dpi=dpi)
            axis = sns.barplot(x=x, y=y, hue=condition_on, 
                              palette=paletter, color=color, ci=conf_intvl)
        else:
            sns.barplot(x=x, y=y, hue=condition_on, ci=conf_intvl, 
                        palette=paletter, color=color, ax=axis)
        
        axis.set_title(plot_title, weight='bold', size=15, x=0.5, y=1.05)
        
        if rotate_xticklabe:
                axis.set_xticklabels(axis.get_xticklabels(), rotation=90)
        if x_labe:
            axis.set_xlabel(str.capitalize(x_labe), weight='bold')
        if y_labe:
            axis.set_ylabel(str.capitalize(y_labe), weight='bold')
        
        y_range = y.max() - y.min()
        if not bot_labe_gap:
            bot_labe_gap=y_range/1000
        if not top_labe_gap:
            top_labe_gap=y_range/30
        
        if annotate:    
            for i in range(len(freq_col)):
                    labe = freq_col.iloc[i]
                    axis.text(i-h_labe_shift, freq_col.iloc[i]+bot_labe_gap, labe,
                              color=bot_labe_color, size=annot_size, weight='bold')
                    if include_perc and perc_freq is not None:
                        labe = f'({perc_freq.iloc[i]}%)'
                        axis.text(i-h_labe_shift, freq_col.iloc[i]+top_labe_gap, labe,
                                  color=top_labe_color, size=annot_size, weight='bold')
                              
        if savefig:
            print(cls.fig_writer(fig_filename, fig))
        return axis
        
    @staticmethod
    def plot_bar(x, y, condition_on=None, plot_title='A Bar Chart', x_labe=None, y_labe=None,
             color=None, paletter='viridis', conf_intvl=None, include_perc=False, perc_freq=None, annot_size=6,
             bot_labe_gap=10, top_labe_gap=10, v_labe_shift=0.4, top_labe_color='black', 
             bot_labe_color='blue', index_order: bool=True, rotate_yticklabe=False, annotate=False, axis=None,
             figsize=(8, 4), dpi=150):
        """plot bar graph on an axis.
        If include_perc is True, then perc_freq must be provided.
        :Return: axis """

        freq_col = x

        if color:
            paletter = None

        if not axis:
            plt.figure(figsize=figsize, dpi=dpi)
            axis = sns.barplot(x=x, y=y, hue=condition_on, orient='h',
                              palette=paletter, color=color, ci=conf_intvl)
        else:
            sns.barplot(x=x, y=y, hue=condition_on, ci=conf_intvl, 
                        palette=paletter, color=color, orient='h', ax=axis)

        axis.set_title(plot_title, weight='bold', size=15, x=0.5, y=1.05)

        if rotate_yticklabe:
            axis.set_yticklabels(axis.get_yticklabels(), rotation=90)
        if x_labe:
            axis.set_xlabel(str.capitalize(x_labe), weight='bold')
        if y_labe:
            axis.set_ylabel(str.capitalize(y_labe), weight='bold')
        if annotate:
            for i in range(len(freq_col)):
                labe = freq_col.iloc[i]
                axis.text(freq_col.iloc[i]+bot_labe_gap, i-v_labe_shift, labe,
                          color=bot_labe_color, size=annot_size, weight='bold')
                if include_perc and perc_freq is not None:
                    labe = f'({perc_freq.iloc[i]}%)'
                    axis.text(freq_col.iloc[i]+top_labe_gap, i-v_labe_shift, labe,
                              color=top_labe_color, size=annot_size, weight='bold')
        return axis
    
    @staticmethod
    def plot_box(x=None, y=None, condition_on=None, plot_title="A Boxplot", orientation='horizontal', 
                 x_labe=None, y_labe=None, axis=None, figsize=(8, 4), dpi=150, savefig=False, fig_filename='boxplot.png'):
        """A box distribution plot on an axis.
        Return: axis"""
        
        if (not isinstance(x, (pd.Series, np.ndarray)) and x is not None):
            raise TypeError("x must be a pandas series or numpy array")
        elif (not isinstance(y, (pd.Series, np.ndarray)) and y is not None):
            raise TypeError("y must be a pandas series or numpy array")
            
        if not axis:
            plt.figure(figsize=figsize, dpi=dpi)
            axis = sns.boxplot(x=x, y=y, hue=condition_on, orient=orientation)
        else:
            sns.boxplot(x=x, y=y, hue=condition_on, orient=orientation, ax=axis)
        axis.set_title(plot_title, weight='bold', x=0.5, y=1.025)
        if x_labe:
                axis.set_xlabel(str.capitalize(x_labe), weight='bold')
        if y_labe:
            axis.set_ylabel(str.capitalize(y_labe), weight='bold')
            
        if savefig:
            print(cls.fig_writer(fig_filename, fig))
        return axis
    
    @staticmethod
    def plot_freq(data: 'DataFrame or Series', freq_col_name: str=None,  plot_title: str='Frequency Plot', 
                  include_perc=False, annot_size=6, top_labe_gap=None, bot_labe_gap=None, h_labe_shift=0.4, 
                  index_order: bool=True, fig_h: int=6, fig_w=4, dpi=150, x_labe=None, y_labe=None, color=None, 
                  rotate_xticklabe=False, axis=None, savefig=False, fig_filename='freqplot.png'):
        """plot bar chart on an axis using a frequecy table
        :Return: axis"""
        
        if not isinstance(data, (pd.DataFrame, pd.Series)):
            raise TypeError("Data must be a dataframe or series")
            
        if isinstance(data, pd.Series):
            freq_col = data.value_counts().sort_index()
        
        elif isinstance(data, pd.DataFrame):
            freq_col = data[freq_col_name].value_counts().sort_index()

        paletter = 'viridis'
        if color:
            paletter = None
        
        if not index_order:
            freq_col = freq_col.sort_values()
            
        if include_perc:
            perc_freq = np.round(100 * freq_col/len(data), 2)
        
        if not axis:
            plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
            axis = sns.barplot(x=freq_col.index, y=freq_col, palette=paletter, color=color)
        else:
            sns.barplot(x=freq_col.index, y=freq_col, palette=paletter, color=color, ax=axis)
        
        y_range = freq_col.max() - freq_col.min()
        if not bot_labe_gap:
            bot_labe_gap=y_range/1000
        if not top_labe_gap:
            top_labe_gap=y_range/30
            
        for i in range(len(freq_col)):
            labe = freq_col.iloc[i]
            axis.text(i-h_labe_shift, freq_col.iloc[i]+bot_labe_gap, labe,
                   size=annot_size, weight='bold')
            if include_perc:
                labe = f'({perc_freq.iloc[i]}%)'
                axis.text(i-h_labe_shift, freq_col.iloc[i]+top_labe_gap, labe,
                       color='blue', size=annot_size, weight='bold')
        if x_labe:
            axis.set_xlabel(str.capitalize(x_labe), weight='bold')
        else:
            axis.set_xlabel(str.capitalize(freq_col.name))
        if y_labe:
            axis.set_ylabel(str.capitalize(y_labe), weight='bold') 
        else:
            axis.set_ylabel('Count')
        
        if rotate_xticklabe:
            axis.set_xticklabels(axis.get_xticklabels(), rotation=90)
        axis.set_title(plot_title, weight='bold', size=15, x=0.5, y=1.025)
        
        if savefig:
            print(cls.fig_writer(fig_filename, fig))
        return axis 
    
    @staticmethod
    def split_date_series(date_col, sep='/', year_first=True):
        """split series containing date data in str format into
        dataframe of tdayee columns (year, month, day)
        Return:
        Dataframe"""
        
        date_col = date_col.str.split(sep, expand=True)
        if year_first:
            day = date_col[2]
            mon = date_col[1]
            yr = date_col[0]
        else:
            day = date_col[0]
            mon = date_col[1]
            yr = date_col[2]
        
        return pd.DataFrame({'Year': yr,
                            'Month': mon,
                            'Day': day})
                            
    @staticmethod
    def plot_correl_heatmap(df, plot_title, title_size=10, annot_size=6, xticklabe_size=6, yticklabe_size=6,
                        xlabe=None, ylabe=None, axlabe_size=8, run_correlation=True, axis=None):
        """plot correlation heatmap from dataframe"""
        
        if run_correlation:
            corr_vals = np.round(df.corr(), 2)
        else:
            corr_vals = df
            
        if not axis:
            axis = sns.heatmap(corr_vals,  fmt='0.2f',
                               annot_kws={'fontsize':annot_size}, annot=True)#, square=True,)
        else:
            sns.heatmap(corr_vals, ax=axis, fmt='0.2f',
                               annot_kws={'fontsize':annot_size}, annot=True)#, square=True,)
                         
        axis.set_title(plot_title, size=title_size, weight='bold', x=0.5, y=1.05)
        
        axis.set_xticklabels(axis.get_xticklabels(), size=xticklabe_size)
        axis.set_yticklabels(axis.get_yticklabels(), size=yticklabe_size)
        
        if xlabe:
            axis.set_xlabel(str.capitalize(xlabe), weight='bold', size=axlabe_size)
        if ylabe:
            axis.set_ylabel(str.capitalize(ylabe), weight='bold', size=axlabe_size)
        
        return axis
        
    @staticmethod
    def visualize_trends(database: pd.DataFrame, fig_filename_suffix='fig'):
        """plots showing accident counts per month, week, day, day_name, hour, and minute;
        comparison plots"""
        
        cls = RoadAccidents()
        def plot_trend_line(x, y, plot_title='Line Plot', title_size=10, x_labe='x axis', y_labe='y axis',
                            text_size=6, text_color='white', bbox_color='black', fig_file_name='line_plot.png', 
                             xlim=(0, 10), fig_w=6, fig_h=8):
        
            # set background style
            sns.set_style('darkgrid')
            # compute average 
            avg = y.mean().round(2)
            # boundaries for average line marker
            avg_line_pos=(xlim[0]-2, xlim[1]+2)

            # figure dimensions
            fig = plt.figure(figsize=(fig_w, fig_h), dpi=200)
            # average line
            cls.plot_line([avg_line_pos[0], avg_line_pos[1]], [avg, avg], color='black', line_size=1)
            # trend line
            cls.plot_line(x, y, marker='o', color='red', 
                           y_labe=y_labe, x_labe=x_labe)
            x_range = range(len(x))
            plt.xlim(left=xlim[0], right=xlim[1])
            # legend for average line
            labe = f'AVERAGE LINE: {avg}'
            text_loc_x, text_loc_y = (len(x)/2)-0.5, avg
            
            plt.text(text_loc_x, text_loc_y, labe, color=text_color, size=text_size, 
                     weight='bold', bbox={'facecolor':bbox_color})
            plt.title(plot_title, size=title_size, weight='bold')
            plt.show()
            # save figure to filesystem as png
            fname = fig_file_name
            print(cls.fig_writer(fname, fig))
            
        def plot_distribution(x, y, plot_title='Line Plot', x_labe='x axis', y_labe='y axis', include_perc=False,
                              text_size=6, text_color='white', bbox_color='black', fig_file_name='distribution_plot.png', 
                               xlim=(0, 10), fig_w=6, fig_h=8):
            "plot accident distribution across hour of day."
            
            # set background style
            sns.set_style('darkgrid')
            
            avg = y.mean().round(2)
            # boundaries of average line marker
            avg_line_pos=(xlim[0]-2, xlim[1]+2)
            # color_mapping: red for above average, orange for below average
            color_mapping = {i: 'red' if v > avg else 'gray' for i, v in zip(x, y)}

            labe = f'AVERAGE LINE: {avg}'
            # figure dimensions
            fig = plt.figure(figsize=(fig_w, fig_h), dpi=200)
            # average line
            ax1 = cls.plot_line([avg_line_pos[0], avg_line_pos[1]], [avg, avg], color='black', line_size=1)
            # frequency plot
            y_range = y.max() - y.min()
            if include_perc:
                perc_freq = (100*y/sum(y)).round(2)
                cls.plot_column(x, y, plot_title=plot_title, include_perc=include_perc,
                             perc_freq=perc_freq, bot_labe_gap=y_range/100, top_labe_gap=y_range/30,  
                                 paletter=color_mapping,x_labe='Hour', y_labe='Total Count', axis=ax1)
            else:    
                cls.plot_column(x, y, plot_title=plot_title, include_perc=include_perc,
                                 paletter=color_mapping,x_labe='Hour', bot_labe_gap=y_range/100, top_labe_gap=y_range/30, 
                                 y_labe='Total Count', axis=ax1)

#             x_range = range(len(x))
            ax1.set_xlim(left=xlim[0], right=xlim[1])
            
            # legend for average line
            labe = f'AVERAGE LINE: {avg}'
            text_loc_x, text_loc_y = (len(x)/2) - 0.5, avg
            ax1.text(text_loc_x, text_loc_y, labe, color=text_color, size=text_size, 
                     weight='bold', bbox={'facecolor':bbox_color})
            plt.show()
            # save figure to filesystem as png
            fname = fig_file_name
            print(cls.fig_writer(fname, fig))
            
        def plot_pyramid(negative_side, positive_side, left_legend='Left', right_legend='right',
                         plot_title='Pyramid Plot', left_legend_color='white', right_legend_color='white', 
                         l_bbox_color='orange', r_bbox_color='blue', fig_w=6, fig_h=8, fig_file_name='pyramid_plot.png'):
            """Pyramid view of negative values vs positive values."""
            
    #         color_mapping = {i: 'blue' if v >= 0 else 'orange' for i, v in enumerate(joined)}
            # print(color_mapping)
            fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=200)

            cls.plot_bar(x=negative_side, y=negative_side.index, 
                           y_labe='Hour', x_labe='Total Accidents', bot_labe_color='black',
                           color=l_bbox_color, annotate=True, v_labe_shift=0.05, axis=ax)
            
            cls.plot_bar(x=positive_side, y=positive_side.index, plot_title=plot_title,
                           y_labe='Hour', x_labe='Total Accidents', bot_labe_color='black',
                           color=r_bbox_color, annotate=True, v_labe_shift=0.05, axis=ax)

            neg_range = (abs(negative_side.min()) - abs(negative_side.max()))
            pos_range = positive_side.max() - positive_side.min()
            left_pos = negative_side.min() - neg_range/2
            right_pos = positive_side.max() + pos_range/2
            
            xlim = (left_pos, right_pos)
            ax.set_xlim(xlim[0], xlim[1])
            
            labe = left_legend
            min_ind = negative_side.loc[negative_side == negative_side.max()].index[0]
            x_pos = (abs(negative_side.min()) - abs(negative_side.max()))/2
            ax.text(-x_pos, min_ind, labe, color=left_legend_color, 
                    weight='bold', bbox={'facecolor':l_bbox_color})
            
            labe = right_legend
            max_ind = positive_side.loc[positive_side == positive_side.min()].index[0]
            x_pos = (positive_side.max() - positive_side.min())/2
            ax.text(x_pos, max_ind, labe, color=right_legend_color, 
                    weight='bold', bbox={'facecolor':r_bbox_color})
            plt.show()
            # save figure to filesystem as png
            fname = fig_file_name
            print(cls.fig_writer(fname, fig))
            
        def plot_diff(left_side, right_side, left_legend='Left', right_legend='right',
                         plot_title='Comparison Plot', left_legend_color='white', right_legend_color='white', 
                      l_bbox_color='orange', r_bbox_color='blue', fig_w=6, fig_h=8, fig_file_name='comparison_plot.png'):
            """Comparison view of left values vs right values."""
            
            diff = right_side - left_side
            color_mapping = {i: r_bbox_color if v >= 0 else l_bbox_color for i, v in zip(diff.index, diff)}
    #         print(color_mapping)
            fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=200)

            cls.plot_bar(x=diff, y=diff.index, y_labe='Hour', plot_title=plot_title, x_labe='Total Accidents',
                          bot_labe_color='black', annotate=True, v_labe_shift=0.05, paletter=color_mapping, 
                          axis=ax)

            left_pos = diff.min() + diff.min()/2
            right_pos = diff.max() + diff.max()/2
            
            xlim = (left_pos, right_pos)
            ax.set_xlim(xlim[0], xlim[1])
            
            labe = left_legend
            min_ind = left_side.loc[left_side == left_side.min()].index[0]
            x_pos = diff.min()/2
            ax.text(x_pos, min_ind, labe, color=left_legend_color, 
                    weight='bold', bbox={'facecolor':l_bbox_color})
            
            labe = right_legend
            min_ind = right_side.loc[right_side == right_side.min()].index[0]
            x_pos = diff.max()/2
            ax.text(x_pos, min_ind, labe, color=right_legend_color, 
                    weight='bold', bbox={'facecolor':r_bbox_color})
            plt.show()
            # save figure to filesystem as png
            fname = fig_file_name
            print(cls.fig_writer(fname, fig))
            
        def plot_dow_distribution(left_side, right_side, lplot_title='Line Plot', rplot_title='Line Plot', left_xlabe='x axis', left_ylabe='y axis',
                                  right_xlabe='x axis', right_ylabe='y axis', include_perc=False, text_size=6, 
                                  text_color='white', bbox_color='black', fig_file_name='distribution_plot.png', 
                                   xlim=(0, 8), fig_w=6, fig_h=8):
            """plot double axes distribution plot for days of week."""
    
            left_x, left_y = left_side['day_name'], left_side['total_count']
            right_x, right_y = right_side['day_name'], right_side['total_count']

            avg = left_y.mean().round(2)
            # boundaries of average line marker
            left_avgline_pos=(xlim[0]-2, xlim[1]+2)
            # color_mapping: red for above average, gray for below average
            color_mapping = {i: 'red' if v >= avg else 'gray' for i, v in zip(left_x, left_y)}

            fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h), dpi=200)
            left, right = axes

            # average line
            cls.plot_line([left_avgline_pos[0], left_avgline_pos[1]], [avg, avg],
                           color='black', line_size=1, axis=left)
            # frequency plot
            y_range = left_y.max() - left_y.min()
            if include_perc:
                perc_freq = (100*left_y/sum(left_y)).round(2)
                cls.plot_column(left_x, left_y, plot_title=lplot_title, include_perc=include_perc,
                             perc_freq=perc_freq, bot_labe_gap=y_range/100, top_labe_gap=y_range/30,
                                 paletter=color_mapping, x_labe=left_xlabe, y_labe=left_ylabe, axis=left, 
                                 rotate_xticklabe=True)
            else:    
                cls.plot_column(left_x, left_y, plot_title=lplot_title, include_perc=include_perc,
                                 paletter=color_mapping, x_labe=left_xlabe, bot_labe_gap=y_range/100, 
                                 top_labe_gap=y_range/30, y_labe=left_ylabe, axis=left, rotate_xticklabe=True)


            avg2 = right_y.mean().round(2)
            # boundaries of average line marker
            right_avgline_pos=(xlim[0]-2, xlim[1]+2)
            # color_mapping: red for above average, gray for below average
            color_mapping = {i: 'red' if v >= avg2 else 'gray' for i, v in zip(right_x, right_y)}
            cls.plot_line([right_avgline_pos[0], right_avgline_pos[1]], [avg2,  avg2],
                            color='black', line_size=1, axis=right)  #average line marker
            # frequency plot
            y_range = right_y.max() - right_y.min()
            if include_perc:
                perc_freq = (100*right_y/sum(right_y)).round(2)
                cls.plot_column(right_x, right_y, plot_title=rplot_title, include_perc=include_perc,
                             perc_freq=perc_freq, bot_labe_gap=y_range/100, top_labe_gap=y_range/30,
                                 paletter=color_mapping, x_labe=right_xlabe, 
                             y_labe=right_ylabe, axis=right, rotate_xticklabe=True)
            else:    
                cls.plot_column(right_x, right_y, plot_title=rplot_title, include_perc=include_perc,
                                  paletter=color_mapping, x_labe=right_xlabe, y_labe=right_ylabe, axis=right,
                                 bot_labe_gap=y_range/100, top_labe_gap=y_range/30, rotate_xticklabe=True)

            labe = f'AVERAGE LINE: {avg}'
            ltext_pos=(0, avg)
            left.text(ltext_pos[0], ltext_pos[1], labe, color=text_color, size=text_size, 
                      weight='bold', bbox={'facecolor':bbox_color})

            labe = f'AVERAGE LINE: {avg2}'
            rtext_pos=(0, avg2)
            right.text(rtext_pos[0], rtext_pos[1], labe, color=text_color, size=text_size, 
                       weight='bold', bbox={'facecolor':bbox_color})

            plt.show()
            
            # save figure to filesystem as png
            fname = fig_file_name
            print(cls.fig_writer(fname, fig))
            
            
        
        # 1. create frequency table for easier selection
        cols = ['month', 'week_num', 'day_num', 'day_of_week', 'day_name', 'is_weekend', 'hour', 'minute']
        general_agg = cls.generate_aggregated_lookup(database, using_cols=cols)
    #     print(general_agg)
        
        # 2. Monthly Accidents
        cols = ['month', 'total_count']
        total_acc_mn = general_agg[cols].groupby(cols[0]).sum().round(2)
    #     print(total_acc_mn)
        
        # 2b. visualize 
        plot_trend_line(total_acc_mn.index, y=total_acc_mn['total_count'], plot_title=f'{str.capitalize(fig_filename_suffix)} Accidents Trend per Month',
                        x_labe='month', y_labe='total accident', text_size=6, text_color='white',
                       bbox_color='black', fig_file_name=f'accident_trend_month_{fig_filename_suffix.lower()}.png', 
                        xlim=(0, 13))
        
        # 3. Weekly Accidents
        cols = ['week_num', 'total_count']
        total_acc_wk = general_agg[cols].groupby(cols[0]).sum()
    #     print(total_acc_wk)
        
        # 3b. visualize 
        plot_trend_line(total_acc_wk.index, y=total_acc_wk['total_count'], title_size=18,
                        plot_title=f'{str.capitalize(fig_filename_suffix)} Accidents Trend per Week of Year',
                        x_labe='week_num', y_labe='total accident', text_size=6, text_color='white', bbox_color='black',
                        fig_file_name=f'accident_trend_week_{fig_filename_suffix.lower()}.png',
                        xlim=(0, 53), fig_h=6, fig_w=15)
        
        # 4. Daily Accidents
        cols = ['day_num', 'total_count']
        total_acc_dy = general_agg[cols].groupby(cols[0]).sum()
    #     print(total_acc_wk)
        
        # 4b. visualize 
        plot_trend_line(total_acc_dy.index, y=total_acc_dy['total_count'], plot_title=f'{str.capitalize(fig_filename_suffix)} Accidents Trend per Day in Year',
                        title_size=25, x_labe='day_num', y_labe='total accident', text_size=8, text_color='white',
                       bbox_color='black', fig_file_name=f'accident_trend_day_{fig_filename_suffix.lower()}.png', 
                         xlim=(0, 370), fig_h=8, fig_w=20)
        
        # 5. Hourly Accidents
        cols = ['hour', 'total_count']
        total_acc_hr = general_agg[cols].groupby(cols[0]).sum()
    #     print(total_acc_wk)
        
        # 5b. visualize 
        plot_trend_line(total_acc_hr.index, y=total_acc_hr['total_count'], 
                        plot_title=f'{str.capitalize(fig_filename_suffix)} Accidents Trend per Hour',
                        x_labe='hour', y_labe='total accident', text_size=6, text_color='white',
                       bbox_color='black', fig_file_name=f'accident_trend_hour_{fig_filename_suffix.lower()}.png', 
                         xlim=(-0.5, 25))
        
        # 6. Minutely Accidents
        cols = ['minute', 'total_count']
        total_acc_min = general_agg[cols].groupby(cols[0]).sum()
    #     print(total_acc_wk)
        
        # 6b. visualize 
        plot_trend_line(total_acc_min.index, y=total_acc_min['total_count'], plot_title=f'{str.capitalize(fig_filename_suffix)} Accidents Trend per Minute of Hour',
                        x_labe='minute', y_labe='total accident', text_size=6, text_color='white', title_size=25,
                       bbox_color='black', fig_file_name=f'accident_trend_min_{fig_filename_suffix.lower()}.png', 
                         xlim=(0, 60), fig_h=8, fig_w=12)
        
        # 7. visualize frequency of accidents per hour
        plot_distribution(total_acc_hr.index, y=total_acc_hr['total_count'], plot_title=f'{str.capitalize(fig_filename_suffix)} Accidents Trend per Hour',
                        x_labe='hour', y_labe='total accident', text_size=6, text_color='white',
                       bbox_color='black', fig_file_name=f'accident_distribution_hour_{fig_filename_suffix.lower()}.png',
                           xlim=(-1, 24), include_perc=True, fig_w=10, fig_h=6)
        
        # 8. Difference between am vs pm accident count
        am_acc = total_acc_hr.loc[total_acc_hr.reset_index().apply(lambda row: row['hour'] in range(12), 
                                                                        axis=1)].reset_index()
        pm_acc = total_acc_hr.loc[total_acc_hr.reset_index().apply(lambda row: row['hour'] in range(12, 24), 
                                                                   axis=1)].reset_index()
        # 8b. pyramid view
        plot_pyramid(-am_acc['total_count'], pm_acc['total_count'], plot_title=f'{str.capitalize(fig_filename_suffix)} AM VS PM Accidents', left_legend='AM', 
                     right_legend='PM', fig_w=12, fig_file_name=f'am_pm_pyramid_{fig_filename_suffix.lower()}.png')                                                     
        
        # 8c. comparison view
        plot_diff(am_acc['total_count'], pm_acc['total_count'], plot_title=f'{str.capitalize(fig_filename_suffix)} Differences Between AM and PM Accidents', 
                   left_legend='AM', right_legend='PM', fig_file_name=f'am_pm_comparison_{fig_filename_suffix.lower()}.png',
                  fig_w=12)
        
        # 9. Accidents per day of week
        cols = ['day_of_week', 'day_name', 'total_count']
        total_acc_dow = general_agg[cols].groupby(cols[:-1]).sum().reset_index()
        total_acc_dow
        
        # weekly average per day of week
        cols = ['day_of_week', 'day_name', 'week_num', 'total_count']
        avg_acc_dow = general_agg[cols].groupby(cols[:-1]).sum().reset_index().round(2)
        # avg_acc_dow

        cols = ['day_of_week', 'day_name', 'total_count']
        avg_acc_dow = avg_acc_dow[cols].groupby(cols[:-1]).mean().round(2).reset_index()
        avg_acc_dow
        # 9b. visualize
        plot_dow_distribution(total_acc_dow, avg_acc_dow, text_size=8, left_xlabe='day of Week', 
                              left_ylabe='total accident', right_xlabe='day of Week', right_ylabe='Average Count',
                              fig_w=12, fig_h=6,  lplot_title=f'{str.capitalize(fig_filename_suffix)} Total accidents per Days of Week',
                             rplot_title=f'{str.capitalize(fig_filename_suffix)} Average Accidents per Days of Week', 
                              fig_file_name=f'dow_distribution_{fig_filename_suffix.lower()}.png')
        
        
    @staticmethod
    def file_search(search_from: 'path_like_str'=None, search_pattern_in_name: str=None, search_file_type: str=None, print_result: bool=False):
        """
        returns a str containing the full path/location of all the file(s)
        matching the given search pattern and file type
        """
    
        # raise error when invalid arguments are given
        if (search_from is None):
            raise ValueError('Please enter a valid search path')
        if (search_pattern_in_name is None) and (search_file_type is None):
            raise ValueError('Please enter a valid search pattern and/or file type')
        
        search_result = {}
        print(f"Starting search from: {search_from}\n")
        for fpath, folders, files in os.walk(search_from):
            for file in files:
                # when both search pattern and file type are entered
                if (search_file_type is not None) and (search_pattern_in_name is not None):
                    if (search_file_type.split('.')[-1].lower() in file.lower().split('.')[-1]) and \
                            (search_pattern_in_name.lower() in file.lower().split('.')[0]):
                        search_result.setdefault(file, f'{fpath}\\{file}')

                # when file type is entered without any search pattern
                elif (search_pattern_in_name is None) and (search_file_type is not None):
                    # print(search_file_type)
                    if search_file_type.split('.')[-1].lower() in file.lower().split('.')[-1]:
                        search_result.setdefault(file, f'{fpath}\\{file}')    

                # when search pattern is entered without any file type
                elif (search_file_type is None) and (search_pattern_in_name is not None):
                    if search_pattern_in_name.lower() in file.lower().split('.')[0]:
                        search_result.setdefault(file, f'{fpath}\\{file}')
                        
        if print_result:
            for k,v in search_result.items():
                print(f"{k.split('.')[0]} is at {v}")
                
        return search_result
       
    @staticmethod
    def get_correlations(df: pd.DataFrame, y_col: 'str or Series', x_colnames: list=None):
        """get for %correlations between df features and y_col.
        And return absolute values of correlations.
        Return: xy_corrwith (sorted absolute values)"""
        
        if not isinstance(df, (pd.Series, pd.DataFrame)):
            raise ValueError("df must be either a dataframe or series")
            
        if not isinstance(y_col, (str, pd.Series)):
            raise ValueError("y_col must be either str or series")
            
        if isinstance(y_col, str):
            y_col = df[y_col]
        if isinstance(df, pd.Series):
            df = pd.DataFrame(df)
        
        if x_colnames:    
            x = pd.DataFrame(df[x_colnames])
        else:
            x = df
        
        return x.corrwith(y_col).abs().sort_values(ascending=False).apply(lambda x: 100*x).round(2)
        
    @staticmethod
    def corr_with_kbest(X, y):
        """using sklearn.preprocessing.SelectKBest, quantify correlations
        between features in X and y.
        Return: 
        correlation series"""
        
        X = pd.DataFrame(X)
        selector = s_fs.SelectKBest(k='all').fit(X, y)
        return pd.Series(selector.scores_, index=selector.feature_names_in_).sort_values(ascending=False)
        
    @staticmethod
    def remove_entry_where(df, where_val_is=-1):
        """drop entries containing specified val
        Return:
        df: dataframe without entries containing the specified values"""
        
        col_names = df.columns
        df2 = copy.deepcopy(df)
        del_inds = []
        
        for col in col_names:
            del_inds.extend(df2.loc[df2[col] == where_val_is].index)
        return df2.loc[~df2.index.isin(del_inds)]
        
    @staticmethod
    def split_sequence(X, y=None, test_size=0.2):
        """Split sequentially.
        Return
        (X_train, X_test, y_train, y_test)"""
        
        if (not isinstance(X, (pd.Series, pd.DataFrame, np.ndarray))):
            raise TypeError("X must be an array/dataframe/series")
            
        if (y is not None) and not isinstance(y, (pd.Series, pd.DataFrame, np.ndarray)):
            raise TypeError("y must be an array/dataframe/series")
            
        train_size = 1 - test_size
        train_ind = int(round(len(X)*train_size))
        
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X_train = X.iloc[: train_ind]
            X_test = X.iloc[train_ind:]
            
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y_train = y.iloc[: train_ind]
            y_test = y.iloc[train_ind:]
            
        if isinstance(X, np.ndarray):
            X_train = X[:train_ind]
            X_test = X[train_ind:]
            
        if isinstance(y, np.ndarray):
            y_train = y[:train_ind]
            y_test = y[train_ind:]
        
        if y is not None:    
            return (X_train, X_test, y_train, y_test)
        return (X_train, X_test)
        
    @staticmethod
    def basic_regressor_score(X_train, y_train, X_val, y_val, X_test, y_test):
        """determine rmse of a trained
        basic linear regression model
        Return:
        trained_nnet, history"""
        
        def build_nnet(n_feats):
            K.clear_session()
            est = models.Sequential()
            est.add(layers.Dense(n_feats*4, input_shape=[n_feats], activation='relu'))
            est.add(layers.Dropout(0.2))
            est.add(layers.Dense(n_feats*2, activation='relu'))
            est.add(layers.Dropout(0.5))
            est.add(layers.Dense(2, activation='relu'))
            est.add(layers.Dense(1))
            
            est.compile(optimizer='adam',
                       loss='mean_squared_error')
            return est
        
        scaler = s_prep.MinMaxScaler()
        scaler.fit(X_train)
        sc_Xtrain = scaler.transform(X_train)
        sc_Xval = scaler.transform(X_val)
        sc_Xtest = scaler.transform(X_test)
        
        es = callbacks.EarlyStopping(monitor='val_loss', patience=3)
        nnet = build_nnet(X_train.shape[1])
        history = nnet.fit(sc_Xtrain, y_train, epochs=50,
                           validation_data=[sc_Xval, y_val],)

        preds = nnet.predict(sc_Xtest)
        print('Root Mean Squared Error\n'+
              f'{s_mtr.mean_squared_error(y_test, preds, squared=False)}')
        
        return nnet, pd.DataFrame(history)
        
    @staticmethod
    def visualize_nulls(df, include_perc=True, plot_title="Number of Mising Rows in the df DataFrame",
                        annotate=True, annot_size=6, use_bar=False, top_labe_gap=0.01, fig_filename='missing_data.png', 
                        fig_size=(8, 6), dpi=200, savefig=False):
            """plot count plot for null values in df."""
            
            cls = RoadAccidents()
            
            null_cols = cls.null_checker(df, only_nulls=True)
            y_range = null_cols.max() - null_cols.min()
            
            if include_perc:
                perc_nulls = cls.null_checker(df, only_nulls=True, in_perc=True)
            
            
            if not len(null_cols):
                return 'No null values in the dataframe'

            sns.set_style("whitegrid")
            
            fig = plt.figure(figsize=fig_size, dpi=dpi)
            ax = fig.add_axes([0, 0, 1, 1])
            if use_bar:  #use bar chart
                if include_perc:
                    cls.plot_bar(null_cols, null_cols.index, plot_title=plot_title, rotate_yticklabe=False,
                                  y_labe='Column Names', x_labe='Number of Missing Values', include_perc=True, perc_freq=perc_nulls, 
                                  annotate=annotate, annot_size=annot_size, bot_labe_gap=y_range/1000, top_labe_gap=top_labe_gap, 
                                  v_labe_shift=0.1, color='brown', figsize=fig_size, dpi=dpi, axis=ax)
                else:
                    cls.plot_bar(null_cols, null_cols.index, plot_title=plot_title, rotate_yticklabe=False,
                                  y_labe='Column Names', x_labe='Number of Missing Values', annotate=annotate, annot_size=annot_size, 
                                  bot_labe_gap=y_range/100, top_labe_gap=top_labe_gap, color='brown', figsize=fig_size, dpi=dpi, axis=ax)
                plt.xlim(right=null_cols.max()+y_range/2)
            else:  #use column chart
                if include_perc:
                    cls.plot_column(x=null_cols.index, y=null_cols, plot_title=plot_title, rotate_xticklabe=True, 
                                    x_labe='Column Names', y_labe='Number of Missing Values', include_perc=True, perc_freq=perc_nulls, 
                                    annotate=annotate, annot_size=annot_size, bot_labe_gap=y_range/1000, top_labe_gap=y_range/30, 
                                    color='brown', figsize=fig_size, dpi=dpi, axis=ax)
                else:
                    cls.plot_column(x=null_cols.index, y=null_cols, plot_title=plot_title, rotate_xticklabe=True, 
                                    x_labe='Column Names', y_labe='Number of Missing Values', annotate=annotate, annot_size=annot_size, 
                                    bot_labe_gap=y_range/100, color='brown', figsize=fig_size, dpi=dpi, axis=ax)
            
                plt.ylim(top=null_cols.max()+y_range/2)
            plt.show()
            if savefig:
                print(cls.fig_writer(fig_filename, fig))
    
    @staticmethod
    def null_checker(df: pd.DataFrame, in_perc: bool=False, only_nulls=False):
        """return quantity of missing data per dataframe column."""
        
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Argument you pass as df is not a pandas dataframe")
            
        null_df = df.isnull().sum().sort_values(ascending=True)
            
        if in_perc:
            null_df = (null_df*100)/len(df)
            if len(null_df):
                print("\nIn percentage")
        
        if only_nulls:
            null_df = null_df.loc[null_df > 0]
            if len(null_df):
                print("\nOnly columns with null values are included")
            
        return np.round(null_df, 2)
    
    @staticmethod
    def unique_categs(df: pd.DataFrame):
        """return unique values per dataframe column."""
        
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Argument you pass as df is not a pandas dataframe")
            
        cols = tuple(df.columns)
        uniq_vals = dict()
        for i in range(len(cols)):
            uniq_vals[cols[i]] = list(df[cols[i]].unique())
        return uniq_vals
    
    @staticmethod
    def round_up_num(decimal: str):
        """round up to nearest whole number if preceding number is 5 or greater."""

        whole, dec = str(float(decimal)).split('.')
        return str(int(whole) + 1) if int(dec[0]) >= 5 or (int(dec[0]) + 1 >= 5 and int(dec[1]) >= 5) else whole

    @staticmethod
    def transform_val(ser: pd.Series, guide: dict):
        """Change values in a series from one format to new format specified in the guide dictionary."""
        
        for k, v in guide.items():
            if isinstance(k, (list, tuple)):
                raise TypeError("Only a dictionary of strings is accepted as keys in the guide arg")

        return ser.apply(lambda val: str(guide[val]) if val in guide.keys() else val)

    @staticmethod
    def check_for_empty_str(df: pd.DataFrame):
        """Return True for column containing '' or ' '.
        Output is a dictionary with column name as key and True/False as value."""

        if not isinstance(df, pd.DataFrame):
            raise TypeError("Argument you pass as df is not a pandas dataframe")

        cols = list(df.columns)  # list of columns
        result = dict()
        for i in range(len(cols)):
            # True for columns having empty strings
            result[cols[i]] = df.loc[(df[cols[i]] == ' ') |
                                    (df[cols[i]] == '')].shape[0] > 0
        
        result = pd.Series(result)
        
        return result.loc[result == True]
    
    @staticmethod
    def col_blank_rows(df: pd.DataFrame):
        """check an entire dataframe and return dict with columns containing blank rows 
        (as keys) and a list of index of blank rows (values)."""
      
        cls = RoadAccidents()
        blank_cols = cls.check_for_empty_str(df).index
        #blank_cols = guide.loc[guide[col for col, is_blank in guide.items() if is_blank]
        result = dict()
        for i in range(len(blank_cols)):
            result[blank_cols[i]] = list(df[blank_cols[i]].loc[(df[blank_cols[i]] == " ") |
                                                          (df[blank_cols[i]] == "")].index)
        return result
    
    @staticmethod
    def fig_writer(fname: str, plotter: plt.figure=None, dpi: int=200, file_type='png'):
        """save an image of the given figure plot in the filesystem."""
        
        cls = RoadAccidents()
        plotter.get_figure().savefig(f"{cls.app_folder_loc}\\{fname}", dpi=dpi, format=file_type,
                                         bbox_inches='tight', pad_inches=0.25)
        return fname