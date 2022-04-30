import pickle
import string
import copy
import seaborn as sns
from seaborn.utils import os, np, plt, pd
from sklearn import preprocessing as s_prep, metrics as s_mtr, feature_selection as s_fs, pipeline as s_pipe
from sklearn import utils as s_utils
from tensorflow.keras import models, layers, backend as K, callbacks, metrics as k_mtr, optimizers
from scipy import stats
from mlxtend import frequent_patterns


class RoadAccidents:
    
    def __init__(self):
        os.makedirs(self.app_folder_loc, exist_ok=True)  # create folder 'Files' at current working directory for storing useful files
        self.ssn_cmap = {'autumn': 'black', 'spring': 'green', 'summer': 'yellow', 'winter':'gray'}
        self.dayname_cmap = {'Sunday': 'green', 'Monday': 'blue', 'Tuesday': 'yellow', 'Wednesday': 'darkorange', 'Thursday': 'red', 'Friday': 'black', 'Saturday': 'gray'}
        self.quarter_cmap = {1:'orange', 2:'green', 3:'yellow', 4:'red'}
        self.severity_cmap = {'fatal':'black', 'serious':'red', 'minor':'gray'}
        self.part_of_day_cmap = {'morning':'green', 'afternoon':'yellow', 'evening':'gray', 'night':'black'}
        
    @property
    def app_name(self):
        return 'RoadAccidents'
        
    @staticmethod
    def predict_accident_severity(accidents):
        """input accident records from merged"""
        
        def preprocess_input(accidents):
            candidate_variables = ['location_easting_osgr', 'location_northing_osgr', 'police_force',  'day_of_week',
                                 'local_authority_district', 'local_authority_highway', '1st_road_class', '1st_road_number', 
                                   'road_type', 'speed_limit', 'junction_detail', 'junction_control', '2nd_road_class', 
                                   '2nd_road_number','pedestrian_crossing_human_control', 'pedestrian_crossing_physical_facilities', 
                                   'light_conditions', 'weather_conditions', 'road_surface_conditions', 'special_conditions_at_site',
                                   'carriageway_hazards', 'urban_or_rural_area', 'lsoa_of_accident_location', 'vehicle_manoeuvre', 
                                   'vehicle_location_restricted_lane', 'junction_location', 'vehicle_leaving_carriageway', 'was_vehicle_left_hand_drive',
                                'journey_purpose_of_driver', 'sex_of_driver', 'age_of_driver', 'age_band_of_driver', 'engine_capacity_cc', 'propulsion_code',
                                 'age_of_vehicle', 'driver_imd_decile', 'driver_home_area_type', 'vehicle_imd_decile', 'casualty_class', 'sex_of_casualty',
                                   'age_of_casualty', 'age_band_of_casualty', 'pedestrian_location', 'pedestrian_movement', 'car_passenger', 
                                   'bus_or_coach_passenger', 'pedestrian_road_maintenance_worker', 'casualty_type', 'casualty_home_area_type', 
                                   'casualty_imd_decile','month', 'day', 'quarter', 'week_num','day_num', 'is_weekend', 'season', 'is_dst', 'is_offseason']
            
           
            X = accidents[candidate_variables]
            
            # integer encode highway
            hw = tuple(X['local_authority_highway'].unique())
            higway_codes = {h:n for n, h in zip(range(1, len(hw) + 1), hw)}
            X.loc[:, 'local_authority_highway'] = X['local_authority_highway'].map(higway_codes)

            # integer encode lsoa_of_location
            lsoa = tuple(X['lsoa_of_accident_location'].unique())
            lsoa_codes = {h:n for n, h in zip(range(1, len(lsoa) + 1), lsoa)}
            X.loc[:, 'lsoa_of_accident_location'] = X['lsoa_of_accident_location'].map(lsoa_codes)
        
            print(f"\nSelected accident variables: {tuple(X.columns)}")
            return X
        
        def load_pca(fname):
            
            with open(fname, 'rb') as f:
                obj = pickle.load(f)
            return obj
        
            
        # load model from fs
        fname = 'accident_severity_classifier.h5'
        as_nnet = models.load_model(fname, compile=False)
        
        # preprocess accidents data
        X = preprocess_input(accidents)
        
        # compress X with PCA to 95% explained variance
        fname = 'X_severity_PCA.pkl'
        pca = load_pca(fname)
        x_pca = pca.transform(X)
        
        # predict from compressed x
        preds = as_nnet.predict(x_pca)
        
        return preds
        
    @staticmethod
    def show_layer_shapes(nn_model):
        for i in range(len(nn_model.layers)):

            print(f'Layer {i}: \nInput_shape: {nn_model.layers[i].input_shape}' +
                 f'\nOutput shape: {nn_model.layers[i].output_shape}\n\n')
                 
     @staticmethod
    def compute_balanced_weights(y: 'array', as_samp_weights=True):
        """compute balanced sample weights for unbalanced classes.
        idea is from sklearn:
        balanced_weight = total_samples / (no of classes * count_per_class)
        WHERE:
        total_samples = len(y)
        no of classes = len(np.unique(y))
        no of samples per class = pd.Series(y).value_counts().sort_index().values
        unique_weights = no of classes * no of samples per class
        samp_weights = {labe:weight for labe, weight in zip(np.unique(y), unique_weights)}
        
        weight_per_labe = np.vectorize(lambda l, weight_dict: weight_dict[l])(y, samp_weights)
        Return:
        weight_per_labe: if samp_weights is True
        class_weights: if samp_weights is False"""
        
        y = np.array(y)
        n_samples = len(y)
        n_classes = len(np.unique(y))
        samples_per_class = pd.Series(y).value_counts().sort_index(ascending=True).values
        denom = samples_per_class * n_classes
        unique_weights = n_samples/denom
        cls_weights = {l:w for l, w in zip(np.unique(y), unique_weights)}
        
        if as_samp_weights:
            return np.vectorize(lambda l, weight_dict: weight_dict[l])(y, cls_weights)
        return cls_weights
        
    @staticmethod
    def dataset_split(x_array: 'np.ndarray', y_array: 'np.ndarray', perc_test=0.25, perc_val=None):
        """create train, validation, test sets from x and y dataframes
        Returns:
        if  val_len != 0:  # perc_val is not 0
            (x_train, x_val, x_test, y_train, y_val, y_test)
        else:  # perc_val is 0 (no validation set)
            (x_train, x_test, y_train, y_test)"""

        if not (isinstance(x_array, np.ndarray) or not isinstance(y_array, np.ndarray)):
            raise TypeError("x_array/y_array is not np.ndarray")

        nx, ny = len(x_array), len(y_array)

        if nx != ny:
            raise ValueError("x_array and y_array have unequal number of samples")
            
        if perc_val is None:
            perc_val = 0

        # number of samples for each set
        combo_len = int(nx * perc_test)
        train_len = int(nx - combo_len)
        val_len = int(combo_len * perc_val)
        test_len = combo_len - val_len

        print(f"Training: {train_len} samples")
        print(f"Validation: {val_len} samples")
        print(f"Test: {test_len} samples")

        if sum([train_len, test_len, val_len]) != nx:
            print("Error in computation")
            return None

        # indexes for x_array/y_array
        inds = np.arange(nx)
        np.random.shuffle(inds)

        # random indexes of test, val and training sets
        train_ind = inds[: train_len]
        combo_ind = inds[train_len:]
        
        test_ind = combo_ind[: test_len]
        val_ind = combo_ind[test_len: ]
            
        x_test, y_test = x_array[test_ind], y_array[test_ind]
        x_val, y_val = x_array[val_ind], y_array[val_ind]
        x_train, y_train = x_array[train_ind], y_array[train_ind]

        return (x_train, x_val, x_test, y_train, y_val, y_test) if val_len else (x_train, x_test, y_train, y_test)
    
    @staticmethod
    def build_nn(n_feats, output_units, output_actvn=None, loss=None, metric=None, lr=1.8e-4):
        print(f"Input features: {n_feats}\nLearining Rate: {lr}\n\n")
        
        K.clear_session()
        nn = models.Sequential()
        nn.add(layers.Dense(n_feats*10, input_shape=[n_feats], activation='relu'))
        
        nn.add(layers.Dense(n_feats*5, activation='relu'))
        nn.add(layers.Dropout(0.5))
        
        nn.add(layers.Dense(n_feats*2, activation='relu'))
        nn.add(layers.Dropout(0.5))
        
        
        nn.add(layers.Dense(units=output_units, activation=output_actvn))
        
        nn.compile(optimizer=optimizers.Adam(learning_rate=lr),
                  loss=[loss])
        
        nn.summary()
        return nn
        
    @staticmethod
    def compute_balanced_class_weights(y, class_labels=None):
        """compute a balanced weight per class
        use the output dictionary as argument for class_weight 
        parameter for keras models
        Returns:
        class_weight_coeffs: list of balanced class weights coefficient"""
        
        if class_labels is None:
            class_labels = [0, 1]
            
        cls_weights = s_utils.class_weight.compute_class_weight(class_weight='balanced', classes=class_labels, y=y)
        class_w = {l:w for l, w in zip(class_labels, cls_weights)}
        return class_w
        
    @staticmethod
    def save_python_obj(fname, py_obj):
        """save python object to file system.
        NOTE: always add .pkl to fname"""
        
        with open(fname, 'wb') as f:
            pickle.dump(txt, f)
        print("Python object has been saved")
        
    @staticmethod
    def load_python_obj(fname):
        """"""
        with open(fname, 'rb') as f:
            txt = pickle.load(f)
        print('Loading complete')
        
    @staticmethod
    def generate_multi_index_from_df(df, sort_by_col_num=0):
        """generate a multi-level index series from dataframe unque categories"""
        
        return pd.MultiIndex.from_frame(df, sortorder=sort_by_col_num)
        
    @staticmethod
    def rank_top_occurrences(df, ranking_col=None, top_n=3, min_count_allowed=1):
        """rank top 3 occurrences per given ranking column"""
        
        if not ranking_col:
            ranking_col = list(df.columns)[0]
        counts = df.value_counts().sort_values(ascending=False).reset_index()
        counts = counts.groupby(ranking_col).head(top_n)
        counts.columns = counts.columns.astype(str).str.replace('0', 'total_count')
        if min_count_allowed:
            counts = counts.loc[counts['total_count'] >= min_count_allowed]
        return counts.sort_values('total_count').reset_index(drop=True)
    
    @staticmethod
    def create_label_from_ranking(df, exclude_last_col=True):
    
        if 'total_count' in df.columns:
            df = df.drop('total_count', axis=1)
        ranking_cols = list(df.columns)
        if exclude_last_col:
            ranking_cols = list(df.iloc[:, :-1].columns)
        labe, n_cols = [], len(ranking_cols)
        for i in range(len(df)):
            row_labe = ''
            for j in range(len(ranking_cols)):
                if j == n_cols - 1:
                    row_labe += f"{ranking_cols[j]}: {df.iloc[i, j]}"
                    continue
                row_labe += f"{ranking_cols[j]}: {df.iloc[i, j]} &\n"
                
            labe.append(row_labe)
        return pd.Series(labe, name='combined_variables', index=df.index)
        
    @staticmethod
    def quarterly_impact(agg_df, quarts, day_names, suffix='overall'):
        ### i. Significant day_names per Quarter

        cls = RoadAccidents()
        # Q3 > Q1
        print(f'\n\nQuarter {quarts[0]} > {quarts[1]}')
        cols = ['quarter', 'day_num', 'total_count']
        total_acc_q =  agg_df[cols].groupby(cols[:-1]).sum().reset_index()
        total_acc_q

        big_q = total_acc_q.loc[total_acc_q['quarter'] == quarts[0]]
        display(big_q)
        small_q = total_acc_q.loc[total_acc_q['quarter'] == quarts[1]]
        display(small_q)

        cls.report_a_significance(big_q['total_count'], small_q['total_count'],
                                  X1_name=f'Q{quarts[0]}', X2_name=f'Q{quarts[1]}')
        
        # Dayname1 > Dayname2
        print(f'\n\n{day_names[0]} > {day_names[1]}')
        cols = ['quarter', 'day_of_week', 'day_name', 'total_count']
        total_acc_q =  agg_df[cols].groupby(cols[:-1]).sum().reset_index()
        total_acc_q

        # test for friday quarterly > sunday quarterly
        big_day = total_acc_q.loc[total_acc_q['day_name'] == day_names[0].capitalize()]
        display(big_day)
        small_day = total_acc_q.loc[total_acc_q['day_name'] == day_names[1].capitalize()]
        display(small_day)
        cls.report_a_significance(big_day['total_count'], small_day['total_count'],
                                  X1_name=f'{day_names[0].capitalize()} Quarterly', 
                                   X2_name=f'{day_names[1].capitalize()} Quarterly')

        # weekday quarterly > weekend quarterly
        print(f'\n\nWeekday > Weekend')
        cols = ['quarter', 'is_weekend', 'total_count']
        total_acc_q =  agg_df[cols].groupby(cols[:-1]).sum().reset_index()
        total_acc_q

        wkday_q = total_acc_q.loc[total_acc_q['is_weekend'] == 0]
        display(wkday_q)
        wkend_q = total_acc_q.loc[total_acc_q['is_weekend'] == 1]
        display(wkend_q)

        cls.report_a_significance(wkday_q['total_count'], wkend_q['total_count'],
                                  X1_name='wkday_q', X2_name='wkend_q')
        
    @staticmethod
    def run_apriori(acc_df, use_cols=None, use_colnames=True, min_support=0.5):
        
        sel_cols = acc_df.select_dtypes(exclude='number')
        if use_cols:
            sel_cols = sel_cols[use_cols]
            
        df = frequent_patterns.apriori(pd.get_dummies(sel_cols),
                                       use_colnames=use_colnames, min_support=min_support)
        df['num_sets'] = df['itemsets'].apply(lambda x: len(x))
        return df.sort_values('support', ascending=False)
    
    @staticmethod
    def visualize_3_variable_rlship(accidents, cols,  plot_title='Relationship from occurrences', annotate=True, min_count_allowed=1,
                                annot_size=10, figsize=(10, 18), xlim=None, x_labe=None, y_labe=None, xy_labe_size=12,
                                top_n=3, paletter=None, savefig=False, fig_fname='rlship_plot.png'):
                                
        cls = RoadAccidents()
        catg_df = cls.get_accidents_with_labels(accidents)
        #display(acc_df)
        count = cls.rank_top_occurrences(catg_df[cols], top_n=top_n, min_count_allowed=min_count_allowed)

        x = count[cols[:-1]].apply(lambda row: 
                                   f"{cols[0]}: {row[cols[0]]} &\n{cols[1]} {row[cols[1]]}",
                                  axis=1)
        cls.plot_bar(count['total_count'], x, condition_on=count[cols[-1]], paletter=paletter,
                 figsize=figsize, annotate=annotate, annot_size=annot_size, xlim=xlim,
                 x_labe=x_labe, y_labe=y_labe, xy_labe_size=xy_labe_size, plot_title=plot_title,
                 savefig=savefig, fig_filename=fig_fname)
    
    @staticmethod
    def get_accidents_with_labels(accidents):
        
        cls = RoadAccidents()
        catg_df = cls.get_label_cols(accidents)
        same_cols = [col for col in list(catg_df.columns) if col in accidents.columns]
        #print(same_cols)
        acc1 = accidents.drop(same_cols, axis=1)
        new_acc = pd.concat([acc1, catg_df], axis=1)
        #display(new_acc)
        return new_acc
    
    @property
    def app_folder_loc(self):
        return f'{self.app_name}Output'
        
    @staticmethod
    def assign_home_area(accidents, situation_col='driver_home_area_type'):
            
            guide = {1: 'Urban area', 2: 'Small town', 3: 'Rural', -1: 'Data missing or out of range'}
            
            home_area = accidents[situation_col].map(guide)
            return home_area

    @staticmethod
    def assign_leaving_way(accidents):
            
            guide = {0: 'Did not leave carriageway', 1: 'Nearside', 2: 'Nearside and rebounded', 3: 'Straight ahead at junction', 4: 'Offside on to central reservation',
                     5: 'Offside on to centrl res + rebounded', 6: 'Offside - crossed central reservation', 7: 'Offside', 8: 'Offside and rebounded', -1: 'Data missing or out of range'}
            
            leaving = accidents['vehicle_leaving_carriageway'].map(guide)
            return leaving

    @staticmethod
    def assign_hit_off_way(accidents):
            
            guide = {0: 'None', 1: 'Road sign or traffic signal', 2: 'Lamp post', 3: 'Telegraph or electricity pole', 4: 'Tree', 5: 'Bus stop or bus shelter',
                     6: 'Central crash barrier', 7: 'Near/Offside crash barrier', 8: 'Submerged in water', 9: 'Entered ditch', 10: 'Other permanent object', 
                     11: 'Wall or fence', -1: 'Data missing or out of range'}
            
            hit_off = accidents['hit_object_off_carriageway'].map(guide)
            return hit_off

    @staticmethod
    def assign_first_impact(accidents):
            
            guide = {0: 'Did not impact', 1: 'Front', 2: 'Back', 3: 'Offside', 
                     4: 'Nearside', -1: 'Data missing or out of range'}
            
            impact = accidents['1st_point_of_impact'].map(guide)
            return impact

    @staticmethod
    def assign_junction_loc(accidents):
            
            guide = {0: 'Not at or within 20 metres of junction', 1: 'Approaching junction or waiting/parked at junction approach', 
                     2: 'Cleared junction or waiting/parked at junction exit', 3: 'Leaving roundabout', 4: 'Entering roundabout', 
                     5: 'Leaving main road', 6: 'Entering main road', 7: 'Entering from slip road',
                     8: 'Mid Junction - on roundabout or on main road', -1: 'Data missing or out of range'}
            
            junc_loc = accidents['junction_location'].map(guide)
            return junc_loc
     
    @staticmethod
    def assign_road_class(accidents, situation_col='1st_road_class'):
        
        guide = {1: 'Motorway', 2: 'A(M)', 3: 'A', 4: 'B', 5: 'C', 6: 'Unclassified'}
        r_class = accidents[situation_col].map(guide)
        return r_class

    @staticmethod
    def assign_police_force(accidents):
        
        guide = {1: 'Metropolitan Police', 3: 'Cumbria', 4: 'Lancashire', 5: 'Merseyside', 6: 'Greater Manchester',
                 7: 'Cheshire', 10: 'Northumbria', 11: 'Durham', 12: 'North Yorkshire', 13: 'West Yorkshire', 14: 'South Yorkshire', 
                 16: 'Humberside', 17: 'Cleveland', 20: 'West Midlands', 21: 'Staffordshire', 22: 'West Mercia', 23: 'Warwickshire', 
                 30: 'Derbyshire', 31: 'Nottinghamshire', 32: 'Lincolnshire', 33: 'Leicestershire', 34: 'Northamptonshire', 35: 'Cambridgeshire',
                 36: 'Norfolk', 37: 'Suffolk', 40: 'Bedfordshire', 41: 'Hertfordshire', 42: 'Essex', 43: 'Thames Valley', 44: 'Hampshire', 
                 45: 'Surrey', 46: 'Kent', 47: 'Sussex', 48: 'City of London', 50: 'Devon and Cornwall', 52: 'Avon and Somerset', 
                 53: 'Gloucestershire', 54: 'Wiltshire', 55: 'Dorset', 60: 'North Wales', 61: 'Gwent', 62: 'South Wales', 63: 'Dyfed-Powys', 91: 'Northern', 
                 92: 'Grampian', 93: 'Tayside', 94: 'Fife', 95: 'Lothian and Borders', 96: 'Central', 97: 'Strathclyde', 98: 'Dumfries and Galloway'}
        
        p_force = accidents['police_force'].map(guide)
        return p_force

    @staticmethod
    def assign_jun_detail(accidents):
        
        guide = {0: 'Not at junction or within 20 metres', 1: 'Roundabout', 2: 'Mini-roundabout', 3: 'T or staggered junction', 5: 'Slip road', 
                 6: 'Crossroads', 7: 'More than 4 arms (not roundabout)', 8: 'Private drive or entrance', 9: 'Other junction', -1: 'Data missing or out of range'}
        jun_detail = accidents['junction_detail'].map(guide)
        return jun_detail

    @staticmethod
    def assign_jun_control(accidents):
        
        guide = {0: 'Not at junction or within 20 metres', 1: 'Authorised person', 2: 'Auto traffic signal', 
                 3: 'Stop sign', 4: 'Give way or uncontrolled', -1: 'Data missing or out of range'}
        jun_control = accidents['junction_control'].map(guide)
        return jun_control

    @staticmethod
    def assign_ped_human_cross(accidents):
        
        guide = {0: 'None within 50 metres ', 1: 'Control by school crossing patrol', 
                 2: 'Control by other authorised person', -1: 'Data missing or out of range'}
        ped_human_cross = accidents['pedestrian_crossing_human_control'].map(guide)
        return ped_human_cross

    @staticmethod
    def assign_ped_phy_facilities(accidents):
        
        guide = {0: 'No physical crossing facilities within 50 metres', 1: 'Zebra', 4: 'Pelican, puffin, toucan or similar non-junction pedestrian light crossing',
                 5: 'Pedestrian phase at traffic signal junction', 7: 'Footbridge or subway', 8: 'Central refuge', -1: 'Data missing or out of range'}

        ped_phy_facilities = accidents['pedestrian_crossing_physical_facilities'].map(guide)
        return ped_phy_facilities

    @staticmethod
    def assign_light_conds(accidents):
        
        guide = {1: 'Daylight', 4: 'Darkness - lights lit', 5: 'Darkness - lights unlit', 6: 'Darkness - no lighting', 
                 7: 'Darkness - lighting unknown', -1: 'Data missing or out of range'}

        ligh_conds = accidents['light_conditions'].map(guide)
        return ligh_conds

    @staticmethod
    def assign_cway_hazards(accidents):
        
        guide = {0: 'None', 1: 'Vehicle load on road', 2: 'Other object on road', 3: 'Previous accident', 4: 'Dog on road', 
                 5: 'Other animal on road', 6: 'Pedestrian in carriageway - not injured', 7: 'Any animal in carriageway (except ridden horse)', -1: 'Data missing or out of range'}

        cway_haz = accidents['carriageway_hazards'].map(guide)
        return cway_haz

    @staticmethod
    def assign_urb_rural(accidents):
        
        guide = {1: 'Urban', 2: 'Rural', 3: 'Unallocated'}

        urb_rural = accidents['urban_or_rural_area'].map(guide)
        return urb_rural

    @staticmethod
    def assign_ped_loc(accidents):
        
        guide = {0: 'Not a Pedestrian', 1: 'Crossing on pedestrian crossing facility', 2: 'Crossing in zig-zag approach lines', 3: 'Crossing in zig-zag exit lines', 4: 'Crossing elsewhere within 50m. of pedestrian crossing', 5: 'In carriageway, crossing elsewhere',
                 6: 'On footway or verge', 7: 'On refuge, central island or central reservation', 8: 'In centre of carriageway - not on refuge, island or central reservation', 9: 'In carriageway, not crossing', 10: 'Unknown or other', -1: 'Data missing or out of range'}

        ped_loc = accidents['pedestrian_location'].map(guide)
        return ped_loc

    @staticmethod
    def assign_ped_movt(accidents):
        
        guide = {0: 'Not a Pedestrian', 1: "Crossing from driver's nearside", 2: 'Crossing from nearside - masked by parked or stationary vehicle', 3: "Crossing from driver's offside", 4: 'Crossing from offside - masked by  parked or stationary vehicle', 5: 'In carriageway, stationary - not crossing  (standing or playing)',
                 6: 'In carriageway, stationary - not crossing  (standing or playing) - masked by parked or stationary vehicle', 7: 'Walking along in carriageway, facing traffic', 8: 'Walking along in carriageway, back to traffic', 9: 'Unknown or other', -1: 'Data missing or out of range'}

        ped_movt = accidents['pedestrian_movement'].map(guide)
        return ped_movt

    @staticmethod
    def assign_car_passgr(accidents):
        
        guide = {0: 'Not car passenger', 1: 'Front seat passenger', 2: 'Rear seat passenger', -1: 'Data missing or out of range'}

        car_passgr = accidents['car_passenger'].map(guide)
        return car_passgr

    @staticmethod
    def assign_bus_passgr(accidents):
        
        guide = {0: 'Not a bus or coach passenger', 1: 'Boarding', 2: 'Alighting', 
                 3: 'Standing passenger', 4: 'Seated passenger', -1: 'Data missing or out of range'}

        bus_passgr = accidents['bus_or_coach_passenger'].map(guide)
        return bus_passgr

    @staticmethod
    def assign_ped_worker(accidents):
        
        guide = {0: 'No / Not applicable', 1: 'Yes', 2: 'Not Known', -1: 'Data missing or out of range'}

        ped_worker = accidents['pedestrian_road_maintenance_worker'].map(guide)
        return ped_worker

    @staticmethod
    def assign_cas_type(accidents):
        
        guide = {0: 'Pedestrian', 1: 'Cyclist', 2: 'Motorcycle 50cc and under rider or passenger', 3: 'Motorcycle 125cc and under rider or passenger', 4: 'Motorcycle over 125cc and up to 500cc rider or  passenger',
                 5: 'Motorcycle over 500cc rider or passenger', 8: 'Taxi/Private hire car occupant', 9: 'Car occupant', 10: 'Minibus (8 - 16 passenger seats) occupant', 11: 'Bus or coach occupant (17 or more pass seats)',
                 16: 'Horse rider', 17: 'Agricultural vehicle occupant', 18: 'Tram occupant', 19: 'Van / Goods vehicle (3.5 tonnes mgw or under) occupant', 20: 'Goods vehicle (over 3.5t. and under 7.5t.) occupant', 
                 21: 'Goods vehicle (7.5 tonnes mgw and over) occupant', 22: 'Mobility scooter rider', 23: 'Electric motorcycle rider or passenger', 90: 'Other vehicle occupant', 97: 'Motorcycle - unknown cc rider or passenger', 
                 98: 'Goods vehicle (unknown weight) occupant'}

        cas_type = accidents['casualty_type'].map(guide)
        return cas_type
     
    @staticmethod
    def get_label_cols(accidents):

        cls = RoadAccidents()

        a = cls.assign_casualty_class(accidents)
        b = cls.assign_district(accidents)
        c = cls.assign_engine_propulsion(accidents)
        d = cls.assign_gender(accidents, 'sex_of_casualty')
        e = cls.assign_highway(accidents)
        f = cls.assign_imd_decile(accidents, col='casualty_imd_decile')
        g = cls.assign_part_of_day(accidents)
        h = cls.assign_pl_offseason(accidents)
        i = cls.assign_site_conditions(accidents)
        j = cls.assign_speed_limit(accidents)
        k = cls.assign_vehicle_left_handed(accidents)
        l = cls.assign_vehicle_location(accidents)
        m = cls.assign_vehicle_manoeuvre(accidents)
        n = cls.assign_vehicle_type(accidents)
        o = cls.assign_weather(accidents)
        p = cls.assign_severity(accidents, 'accident_severity')
        q = cls.assign_severity(accidents, 'casualty_severity')
        r = cls.assign_age_band(accidents, 'age_band_of_driver')
        s = cls.assign_age_band(accidents, 'age_band_of_casualty')
        t = cls.assign_gender(accidents, 'sex_of_driver')
        u = cls.assign_imd_decile(accidents, col='driver_imd_decile')
        v = cls.assign_road_type(accidents)
        w = cls.assign_road_surface(accidents)
        x = cls.assign_journey_purpose(accidents)
        y = cls.assign_first_impact(accidents)
        z = cls.assign_home_area(accidents, 'driver_home_area_type')
        ab = cls.assign_home_area(accidents, 'casualty_home_area_type')
        ac = cls.assign_leaving_way(accidents)
        ad = cls.assign_hit_off_way(accidents)
        ae =  cls.assign_junction_loc(accidents)
        af = cls.assign_road_class(accidents, '1st_road_class')
        ag = cls.assign_road_class(accidents, '2nd_road_class')
        ah = cls.assign_urb_rural(accidents)
        ai = cls.assign_police_force(accidents)
        aj = cls.assign_jun_detail(accidents)
        al = cls.assign_jun_control(accidents)
        am = cls.assign_ped_human_cross(accidents)
        an = cls.assign_ped_phy_facilities(accidents)
        ao = cls.assign_light_conds(accidents)
        ap = cls.assign_cway_hazards(accidents)
        aq = cls.assign_urb_rural(accidents)
        ar = cls.assign_ped_loc(accidents)
        at = cls.assign_ped_movt(accidents)
        au = cls.assign_car_passgr(accidents)
        av = cls.assign_bus_passgr(accidents)
        aw = cls.assign_ped_worker(accidents)
        ax = cls.assign_cas_type(accidents)


        cols = [a, b, c, d, e, f, g, h, i, j,
               k, l, m, n, o, p, q, r, s, t, u,
               v, w, x, y, z, ab, ac, ad, ae, af,
               ag, ah, ai, aj, al, am, an, ao, ap, 
               aq, ar, at, au, av, aw, ax]

        acc_df = pd.concat(cols, axis=1)
        #for col in cols:
         #   acc_df[col.name] = col
        #display(acc_df)
        return acc_df
     
    @staticmethod
    def assign_casualty_class(accidents):
        guide = {1: 'Driver or rider', 2: 'Passenger', 3: 'Pedestrian'}
        casualty_class = accidents['casualty_class'].map(guide)
        return casualty_class

    @staticmethod
    def assign_vehicle_manoeuvre(accidents):
        guide = {1: 'Reversing', 2: 'Parked', 3: 'Waiting to go - held up', 4: 'Slowing or stopping', 
                 5: 'Moving off', 6: 'U-turn', 7: 'Turning left', 8: 'Waiting to turn left', 9: 'Turning right', 10: 'Waiting to turn right', 11: 'Changing lane to left', 
                 12: 'Changing lane to right', 13: 'Overtaking moving vehicle - offside', 14: 'Overtaking static vehicle - offside', 15: 'Overtaking - nearside',
                 16: 'Going ahead left-hand bend', 17: 'Going ahead right-hand bend', 
                 18: 'Going ahead other', -1: 'Data missing or out of range'}
        vehicle_manoeuvre = accidents['vehicle_manoeuvre'].map(guide)
        return vehicle_manoeuvre

    @staticmethod
    def assign_gender(accidents, situation_col='sex_of_casualty'):
        guide = {1: 'Male', 2: 'Female', 3: 'Not known', -1: 'Data missing or out of range'}
        gender = accidents[situation_col].map(guide)
        gender.name = situation_col
        return gender

    @staticmethod
    def assign_speed_limit(accidents):
        guide = {20: '20 MPH', 30: '30 MPH', 40: '40 MPH', 50: '50 MPH', 
                 60: '60 MPH', 70: '70 MPH', -1: 'Data missing or out of range'}
        speed_limit = accidents['speed_limit'].map(guide)
        speed_limit.name = 'speed_limit'
        return speed_limit

    @staticmethod
    def assign_site_conditions(accidents):
        guide = {0: 'None', 1: 'Auto traffic signal - out', 2: 'Auto signal part defective', 3: 'Road sign or marking defective or obscured', 4: 'Roadworks', 
                 5: 'Road surface defective', 6: 'Oil or diesel', 7: 'Mud', -1: 'Data missing or out of range'}
        site_conditions = accidents['special_conditions_at_site'].map(guide)
        return site_conditions

    @staticmethod
    def assign_age_band(accidents, situation_col):
        guide = {1: '0 - 5', 2: '6 - 10', 3: '11 - 15', 4: '16 - 20', 5: '21 - 25', 6: '26 - 35',
                 7: '36 - 45', 8: '46 - 55', 9: '56 - 65', 10: '66 - 75', 11: 'Over 75', -1: 'Data missing or out of range'}
        age_band = accidents[situation_col].map(guide)
        age_band.name = situation_col
        return age_band

    @staticmethod
    def assign_vehicle_location(accidents):
        guide = {0: "On main c'way - not in restricted lane", 1: 'Tram/Light rail track', 2: 'Bus lane', 3: 'Busway (including guided busway)', 
                 4: 'Cycle lane (on main carriageway)', 5: 'Cycleway or shared use footway (not part of  main carriageway)', 6: 'On lay-by or hard shoulder',
                 7: 'Entering lay-by or hard shoulder', 8: 'Leaving lay-by or hard shoulder', 9: 'Footway (pavement)', 10: 'Not on carriageway', -1: 'Data missing or out of range'}
        veh_location = accidents['vehicle_location_restricted_lane'].map(guide)
        return veh_location

    @staticmethod
    def assign_road_surface(accidents):
        guide = {1: 'Dry', 2: 'Wet or damp', 3: 'Snow', 4: 'Frost or ice', 5: 'Flood over 3cm. deep', 
                 6: 'Oil or diesel', 7: 'Mud', -1: 'Data missing or out of range'}
        road_surface = accidents['road_surface_conditions'].map(guide)
        return road_surface

    @staticmethod
    def assign_road_type(accidents):
        guide = {1: 'Roundabout', 2: 'One way street', 3: 'Dual carriageway', 6: 'Single carriageway', 7: 'Slip road', 
                 9: 'Unknown', 12: 'One way street/Slip road', -1: 'Data missing or out of range'}
        road_type = accidents['road_type'].map(guide)
        return road_type

    @staticmethod
    def visualize_situation(accidents, situation_col=None, pivot_on='casualty_severity',
                            plot_title='Distribution of Variable', savefig=False, suffix='viz_situation'):
        """visualize frequency of situation columns in accidents."""
        
        cls = RoadAccidents()
        if not situation_col:
            situation_col = 'age_band_of_driver'
        
        x = accidents[situation_col].value_counts(sort=True).sort_index(ascending=True)
            
        if 'sex'  in str.lower(situation_col):
            sex = cls.assign_gender(accidents, situation_col)
            sex.name = situation_col
            x = sex

        elif 'age_band' in str.lower(situation_col):
            age_band = cls.assign_age_band(accidents, situation_col)
            age_band.name = situation_col
            x = age_band

        elif 'speed_limit'  in str.lower(situation_col):
            spd = cls.assign_speed_limit(accidents)
            x = spd
        
        elif 'casualty_class' in str.lower(situation_col):
            cas_class = cls.assign_imd_decile(accidents)
            x = cas_class
            
        elif 'vehicle_manoeuvre' in str.lower(situation_col):
            veh_man = cls.assign_imd_decile(accidents)
            x = veh_man
            
        elif 'special_condition' in str.lower(situation_col):
            site_cond = cls.assign_site_conditions(accidents)
            x = site_cond
            
        elif 'vehicle_location' in str.lower(situation_col):
            v_loc = cls.assign_vehicle_location(accidents)
            x = v_loc
            
        elif 'road_type' in str.lower(situation_col):
            r_type = cls.assign_road_type(accidents)
            x = r_type
            
        elif 'road_surface' in str.lower(situation_col):
            r_surface = cls.assign_road_surface(accidents)
            x = r_surface
        
        severity = cls.assign_severity(accidents)
        severity.name = 'casualty_severity'
        df = pd.concat([x, severity], axis=1)
    #     display(df)
        freq = df.value_counts().reset_index()
        freq.columns = freq.columns.astype(str).str.replace('0', 'total_count')
        freq = freq.sort_values(x.name)
    #     display(freq)
        avg = freq['total_count'].mean()
        cmap = {'fatal':'black', 'serious':'red', 'minor':'gray'}# for i,u in zip(freq[pivot_on], freq['total_count'])}
        unq = list(freq[x.name].unique())
        
        if (len(unq) > 5) and (len(unq) <= 10):
            cls.plot_bar(freq['total_count'], freq[x.name], condition_on=freq[pivot_on], 
                          plot_title=plot_title, annotate=True, figsize=(15, 10), annot_size=12, xy_labe_size=10,
                          x_labe='total_count', y_labe=situation_col, paletter=cmap, savefig=savefig, 
                         fig_filename=f'situation_{situation_col}_{suffix}.png')
        elif (len(unq) <= 5):
            cls.plot_column(freq[x.name], freq['total_count'], condition_on=freq[pivot_on], plot_title=plot_title, 
                             y_labe='total_count', x_labe=situation_col, paletter=cmap, savefig=savefig,
                             fig_filename=f'situation_{situation_col}_{suffix}.png')
        else:
            cls.plot_scatter(y=freq[x.name], x=freq['total_count'], condition_on=freq[pivot_on], plot_title=plot_title, 
                              x_labe='total_count', y_labe=situation_col, paletter=cmap, figsize=(10, 10), savefig=savefig,
                              fig_filename=f'situation_{situation_col}_{suffix}.png')
    @staticmethod
    def visualize_casualty_outcomes_for(accidents, selected_cols=None, savefig=False, suffix='situation'):
        """visualize casualty outcomes for the selected columns"""
        
        cls = RoadAccidents()
        selected_cols = ['age_band_of_driver', 'age_band_of_casualty', 'sex_of_driver', 'sex_of_casualty',
                          'speed_limit', 'casualty_class', 'vehicle_manoeuvre', 'special_conditions_at_site', 'road_type',
                         'road_surface_conditions', 'vehicle_location_restricted_lane',]
        
        for col in selected_cols:
            cls.visualize_situation(accidents, situation_col=col, plot_title=f'Accident Casualty Outcomes per {col}', 
                                    suffix=suffix, savefig=savefig)
     
    @staticmethod
    def mbike_codes():

        guide = {1: 'Pedal cycle', 2: 'Motorcycle 50cc and under', 3: 'Motorcycle 125cc and under', 4: 'Motorcycle over 125cc and up to 500cc', 5: 'Motorcycle over 500cc',
             8: 'Taxi/Private hire car', 9: 'Car', 10: 'Minibus (8 - 16 passenger seats)', 11: 'Bus or coach (17 or more pass seats)', 16: 'Ridden horse', 
             17: 'Agricultural vehicle', 18: 'Tram', 19: 'Van / Goods 3.5 tonnes mgw or under', 20: 'Goods over 3.5t. and under 7.5t', 21: 'Goods 7.5 tonnes mgw and over',
             22: 'Mobility scooter', 23: 'Electric motorcycle', 90: 'Other vehicle', 97: 'Motorcycle - unknown cc', 98: 'Goods vehicle - unknown weight', -1: 'Data missing or out of range'}

        codes = [code for code, veh_type in guide.items() if 'motorcycle' in str.lower(veh_type)]
        return codes
    
    @staticmethod
    def visualize_vehicle_variables(accidents):
    
        cls = RoadAccidents()
        cls.rank_top10(accidents, 'age_of_vehicle', x_labe='age_of_vehicle', y_labe='total_count',
                   savefig=True, fig_fname='top10_veh_age.png')
        
        cls.show_engine_ranking(accidents['engine_capacity_cc'], plot_title='Top Ten Engine Capacity Involved in Accidents',
                       savefig=True, fig_fname='top10_engine.png')
        
        cls.show_propulsion_ranking(accidents, plot_title='Highest Ranking Accidents per Propulsion Type',
                            savefig=True, fig_fname='top10_propulsion.png')
        
        cls.show_lefthand_ranking(accidents, plot_title='Highest Ranking Accidents per Lefthandedness',
                         savefig=True, fig_fname='top10_lefthand.png')
        
        cls.show_vtype_ranking(accidents, plot_title='Highest Ranking Accidents per Vehicle Type',
                      savefig=True, fig_fname='top10_vehicle_types.png')
    
    @staticmethod
    def assign_weather(accidents):
        guide = {1: 'Fine no high winds', 2: 'Raining no high winds', 3: 'Snowing no high winds', 4: 'Fine + high winds', 
                 5: 'Raining + high winds', 6: 'Snowing + high winds', 7: 'Fog or mist', 8: 'Other', 9: 'Unknown', -1: 'Data missing or out of range'}
        
        weather = accidents['weather_conditions'].map(guide)
        return weather
    
    @staticmethod
    def assign_vehicle_type(accidents):
        
        guide = {1: 'Pedal cycle', 2: 'Motorcycle 50cc and under', 3: 'Motorcycle 125cc and under', 4: 'Motorcycle over 125cc and up to 500cc', 5: 'Motorcycle over 500cc', 8: 'Taxi/Private hire car', 9: 'Car', 
                 10: 'Minibus (8 - 16 passenger seats)', 11: 'Bus or coach (17 or more pass seats)', 16: 'Ridden horse', 17: 'Agricultural vehicle', 18: 'Tram', 19: 'Van / Goods 3.5 tonnes mgw or under', 
                 20: 'Goods over 3.5t. and under 7.5t', 21: 'Goods 7.5 tonnes mgw and over', 22: 'Mobility scooter', 23: 'Electric motorcycle', 90: 'Other vehicle', 97: 'Motorcycle - unknown cc', 98: 'Goods vehicle - unknown weight', -1: 'Data missing or out of range'}
        vtype = accidents['vehicle_type'].map(guide)
        return vtype

    @staticmethod 
    def assign_vehicle_left_handed(accidents):
        
        guide = {1: 'No', 2: 'Yes', -1: 'Data missing or out of range'}
        left_hand_drive = accidents['was_vehicle_left_hand_drive'].map(guide)
        return left_hand_drive
      
    @staticmethod
    def assign_engine_propulsion(accidents):
        
        guide = {1: 'Petrol', 2: 'Heavy oil', 3: 'Electric', 4: 'Steam', 5: 'Gas', 6: 'Petrol/Gas (LPG)', 
                 7: 'Gas/Bi-fuel', 8: 'Hybrid electric', 9: 'Gas Diesel', 10: 'New fuel technology', 11: 'Fuel cells',
                 12: 'Electric diesel', 0: 'Undefined'}
        propulsion = accidents['propulsion_code'].map(guide)
        return propulsion

    @staticmethod
    def assign_imd_decile(accidents, col='casualty_imd_decile'):
        
        guide = {1: 'Most deprived 10%', 2: 'More deprived 10-20%', 3: 'More deprived 20-30%', 4: 'More deprived 30-40%', 5: 'More deprived 40-50%',
                 6: 'Less deprived 40-50%', 7: 'Less deprived 30-40%', 8: 'Less deprived 20-30%', 9: 'Less deprived 10-20%', 10: 'Least deprived 10%', -1: 'Data missing or out of range'}
        decile = accidents[col].map(guide)
        return decile

    @staticmethod
    def rank_top10(accidents, col=None, show_all=False, plot_title='Top Ten Accidents', x_labe=None, y_labe=None, 
                   use_bar=False, savefig=False, fig_fname='top10.png'):
        """provide ranking for columns"""
        
        cls = RoadAccidents()
        if isinstance(col, (list, tuple)) and (col is not None):
            result = dict()
            for name in col:
                if show_all:
                    result[name] = accidents[name].value_counts().sort_values(ascending=False)
                else:
                    result[name] = accidents[name].value_counts().sort_values(ascending=False).iloc[:10]
                if use_bar:
                    cls.plot_bar(result[name], result[name].index, annotate=True, plot_title=plot_title,
                                 x_labe=x_labe, y_labe=y_labe, savefig=savefig, fig_filename=fig_fname)
                else:
                    cls.plot_column(result[name].index, result[name], plot_title=plot_title,
                                     x_labe=x_labe, y_labe=y_labe, savefig=savefig, fig_filename=fig_fname)
            return result
        if isinstance(accidents, pd.DataFrame):
            result = accidents[col].value_counts().sort_values(ascending=False).iloc[:10]
            if use_bar:
                cls.plot_bar(result, result.index, annotate=True, plot_title=plot_title,
                             x_labe=x_labe, y_labe=y_labe, savefig=savefig, fig_filename=fig_fname)
            else:
                cls.plot_column(result.index, result, plot_title=plot_title,
                                x_labe=x_labe, y_labe=y_labe, savefig=savefig, fig_filename=fig_fname)
            
            return result
        
        elif isinstance(accidents, pd.Series):
            result = accidents.value_counts().sort_values(ascending=False).iloc[:10]
            if use_bar:
                cls.plot_bar(result, result.index, annotate=True, plot_title=plot_title,
                             x_labe=x_labe, y_labe=y_labe, savefig=savefig, fig_filename=fig_fname)
            else:
                cls.plot_column(result.index, result, plot_title=plot_title,
                                x_labe=x_labe, y_labe=y_labe, savefig=savefig, fig_filename=fig_fname)
            return result

    @staticmethod
    def show_vtype_ranking(accidents, plot_title='Top Ten Accidents', savefig=False, fig_fname='top10_acc.png'):
        
        cls = RoadAccidents()
        vtype = cls.assign_vehicle_type(accidents)
        ranked = cls.rank_top10(vtype)
        cls.plot_bar(y=ranked.index, x=ranked, plot_title=plot_title, annotate=True,
                     savefig=savefig, fig_filename=fig_fname)

    @staticmethod
    def show_propulsion_ranking(accidents, plot_title='Top Ten Accidents', savefig=False, fig_fname='top10_acc.png'):
        
        cls = RoadAccidents()
        eng = cls.assign_engine_propulsion(accidents)
        ranked = cls.rank_top10(eng)
        cls.plot_bar(y=ranked.index, x=ranked, plot_title=plot_title, annotate=True,
                     savefig=savefig, fig_filename=fig_fname)

    @staticmethod
    def show_lefthand_ranking(accidents, plot_title='Top Ten Accidents', savefig=False, fig_fname='top10_acc.png'):
        
        cls = RoadAccidents()
        lhand = cls.assign_vehicle_left_handed(accidents)
        ranked = cls.rank_top10(lhand)
        cls.plot_column(x=ranked.index, y=ranked, plot_title=plot_title,include_perc=True, top_labe_gap=0,
                     h_labe_shift=-0.5, x_labe='left_hand_drive', y_labe='total_count',
                         savefig=savefig, fig_filename=fig_fname)

    @staticmethod
    def show_engine_ranking(engines, plot_title='Top Ten Accidents', savefig=False, fig_fname='top10_acc.png'):
        
        cls = RoadAccidents()
        ranked = cls.rank_top10(engines)
        cls.plot_column(x=ranked.index, y=ranked, plot_title=plot_title,include_perc=True, top_labe_gap=-3000,
                     h_labe_shift=-0.15, x_labe='Engine_capacity', y_labe='total_count', top_labe_color='white',
                         savefig=savefig, fig_filename=fig_fname)
    
    @staticmethod
    def pedestrian_codes():
        guide = {0: 'Pedestrian', 1: 'Cyclist', 2: 'Motorcycle 50cc and under rider or passenger', 3: 'Motorcycle 125cc and under rider or passenger', 
                 4: 'Motorcycle over 125cc and up to 500cc rider or  passenger', 5: 'Motorcycle over 500cc rider or passenger', 8: 'Taxi/Private hire car occupant',
                 9: 'Car occupant', 10: 'Minibus (8 - 16 passenger seats) occupant', 11: 'Bus or coach occupant (17 or more pass seats)', 16: 'Horse rider', 
                 17: 'Agricultural vehicle occupant', 18: 'Tram occupant', 19: 'Van / Goods vehicle (3.5 tonnes mgw or under) occupant', 
                 20: 'Goods vehicle (over 3.5t. and under 7.5t.) occupant', 21: 'Goods vehicle (7.5 tonnes mgw and over) occupant', 22: 'Mobility scooter rider', 
                 23: 'Electric motorcycle rider or passenger', 90: 'Other vehicle occupant', 97: 'Motorcycle - unknown cc rider or passenger', 
                 98: 'Goods vehicle (unknown weight) occupant'}

        codes = [code for code, cas_type in guide.items() if 'pedestrian' in str.lower(cas_type)]
        return codes

    @staticmethod
    def get_mbike_accidents(accidents):
        """select all accidents involving motorbikes/motorcycles."""
        
        cls = RoadAccidents()
        codes = cls.mbike_codes()
        return accidents.loc[accidents['vehicle_type'].isin(codes)]

    @staticmethod
    def get_pedestrian_accidents(accidents):
        """select all accidents involving pedestrians."""
        
        cls = RoadAccidents()
        codes = cls.pedestrian_codes()
        return accidents.loc[accidents['casualty_type'].isin(codes)]
        
    @staticmethod
    def get_daynums_in_wknum(accidents, week_num):
        """display details about a specific week num"""
        
        wknum_accidents = accidents.loc[accidents['week_num'] == week_num, ['day_num', 'month_name', 'day',
                                                                            'season', 'quarter']]
        unq_days = sorted(tuple(wknum_accidents['day_num'].unique()))
        dates = sorted(tuple(wknum_accidents.apply(lambda row: f"{row['day']}-{row['month_name']}",
                                           axis=1).unique()), key=lambda x: int(x.split('-')[0]))
        ssn, qtr, mn = wknum_accidents['season'].unique(), wknum_accidents['quarter'].unique(), wknum_accidents['month_name'].unique()
        print(f"\n\nThere were {len(unq_days)} days in week {week_num} in {mn}.\n"+
             f"The days consisted of days {unq_days}.\n"
              f"Season: {ssn}\nQuarter: {qtr}.\n" +
             f"Dates: {dates}")
        return unq_days
        
    @staticmethod
    def select_geographic_area(accidents, district=None, highway=None):
        """return longitude-latitude coord of given city or district.
        Return: 
        longitude: series containing longitude coords
        latitude: series containing latitude coords"""
        
        cls = RoadAccidents()
        result = pd.DataFrame(accidents)
        if district:
            districts = cls.assign_district(accidents)
            check = districts.apply(lambda x: str.lower(district) in str.lower(x))
            district_index = districts.loc[check].index
            result = result.loc[district_index]
        if highway:
            highways =  cls.assign_highway(accidents)
            check = highways.apply(lambda x: str.lower(highway) in str.lower(x))
            highway_index = highways.loc[check].index
            result = result.loc[highway_index]

        return result
    
    @staticmethod
    def plot_top10_districts(accidents, plot_title='Top Ten Districts for Accidents', title_size=14, background_color='gray',
                            focus_color='red', figsize=(18, 6), dpi=200, savefig=False, suffix='district'):
        
        cls = RoadAccidents()
        districts = cls.assign_district(accidents)
        top10_districts = districts.value_counts().iloc[:10]
        tt_ind = districts.loc[districts.isin(list(top10_districts.index))].index
        df = pd.concat([accidents[['longitude', 'latitude']], districts], axis=1)
        district_accidents = df.loc[tt_ind]
        
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        sns.scatterplot(accidents['longitude'], accidents['latitude'], color=background_color,
                          alpha=None, ax=ax)
        sns.scatterplot(x=district_accidents['longitude'], y=district_accidents['latitude'], hue=district_accidents['local_authority_district'],
                         color=focus_color, ax=ax)
                         
        ax.set_title(plot_title, size=title_size, weight='bold')
        
        if savefig:
            fig_fname=f'top10_{suffix}.png'
            print(cls.fig_writer(fig_fname, fig))
            
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        cls.plot_bar(y=top10_districts.index, x=top10_districts, annotate=True,
                     plot_title=plot_title, annot_size=13, axis=ax)
        
        if savefig:
            fig_fname = f'hist_top10_{suffix}.png'
            print(cls.fig_writer(fig_fname, fig))

        return ax
        

    @staticmethod
    def assign_pl_offseason(accidents):
        """off season was between 13 May & 8 August 2019,
        and on season is outside this period"""

        start_day_num = accidents.loc[(accidents['month_name'] == 'May') &
                                     (accidents['day'] == 13), 'day_num'].values[0]
        end_day_num = accidents.loc[(accidents['month_name'] == 'August') &
                                     (accidents['day'] == 8), 'day_num'].values[0]
        off_season_ix = accidents.loc[accidents['day_num'].between(start_day_num, end_day_num)].index
        off_season = pd.Series(index=accidents.index, name='is_offseason')
        off_season[off_season_ix] = 1
        off_season = off_season.fillna(0).astype(int)
        return off_season
    
    @staticmethod
    def visualize_top_ten_districts(accidents,  col='casualty_severity', select_col_val=1, figsize=(25, 10), dpi=250, suffix='ovr',
                                plot_title='Top Ten Fatal Districts', title_size=20, savefig=False):
        cls = RoadAccidents()
        sns.set_style('darkgrid')
        cols = ['local_authority_district', col]
        distrs = cls.assign_district(accidents)

        df = pd.concat([distrs, accidents[cols[1]]], axis=1)
        
        selected = df.groupby(cols).size().sort_values(ascending=False).reset_index()
        selected.columns = selected.columns.astype(str).str.replace('0', 'total_count')
        if isinstance(select_col_val, (list, tuple)):
            top10_districts = selected.loc[selected[cols[1]].isin(select_col_val)].iloc[10]
        else:
            top10_districts = selected.loc[selected[cols[1]] == select_col_val].iloc[:10]
        
        dds = dict()
        for col in top10_districts[cols[0]]:
            distr = cls.select_geographic_area(accidents, district=col)
            dds[col] = distr[['longitude', 'latitude']]
        
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        sns.scatterplot(x=accidents['longitude'], y=accidents['latitude'], color='gray',# hue=df['part_of_day_num'],
                            ax=ax)
        for col, df in dds.items():
            sns.scatterplot(x=df['longitude'], y=df['latitude'], label=col,# hue=df['part_of_day_num'],
                            ax=ax)
            
        ax.set_title(plot_title, size=title_size, weight='bold')
        
        if savefig:
            fig_fname=f'top10_districts_{suffix}.png'
            print(cls.fig_writer(fig_fname, fig))
            
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        for col, df in dds.items():
            cls.plot_bar(y=top10_districts[cols[0]], x=top10_districts['total_count'], annotate=True,
                            plot_title=plot_title, annot_size=13, axis=ax)
        
        if savefig:
            fig_fname = f'hist_top10_districts_{suffix}.png'
            print(cls.fig_writer(fig_fname, fig))

        return ax
    
    @staticmethod
    def assign_journey_purpose(accidents):
        guide = {1: 'Journey as part of work', 2: 'Commuting to/from work', 3: 'Taking pupil to/from school',
                 4: 'Pupil riding to/from school', 5: 'Other', 6: 'Not known', 15: 'Other/Not known (2005-10)', -1: 'Data missing or out of range'}
        
        purpose = accidents['journey_purpose_of_driver'].map(guide)
        return purpose
    
    @staticmethod
    def assign_district(accidents):
        distr_guide = {1: 'Westminster', 2: 'Camden', 3: 'Islington', 4: 'Hackney', 5: 'Tower Hamlets', 6: 'Greenwich', 7: 'Lewisham', 8: 'Southwark', 9: 'Lambeth', 10: 'Wandsworth', 
                       11: 'Hammersmith and Fulham', 12: 'Kensington and Chelsea', 13: 'Waltham Forest', 14: 'Redbridge', 15: 'Havering', 16: 'Barking and Dagenham', 17: 'Newham', 18: 'Bexley', 19: 'Bromley', 20: 'Croydon',
                       21: 'Sutton', 22: 'Merton', 23: 'Kingston upon Thames', 24: 'Richmond upon Thames', 25: 'Hounslow', 26: 'Hillingdon', 27: 'Ealing', 28: 'Brent', 29: 'Harrow', 30: 'Barnet',
                       31: 'Haringey', 32: 'Enfield', 33: 'Hertsmere', 38: 'Epsom and Ewell', 40: 'Spelthorne', 57: 'London Airport (Heathrow)', 60: 'Allerdale', 61: 'Barrow-in-Furness', 62: 'Carlisle', 63: 'Copeland', 64: 'Eden', 
                       65: 'South Lakeland', 70: 'Blackburn with Darwen', 71: 'Blackpool', 72: 'Burnley', 73: 'Chorley', 74: 'Fylde', 75: 'Hyndburn', 76: 'Lancaster', 77: 'Pendle', 79: 'Preston', 80: 'Ribble Valley', 82: 'Rossendale',
                       83: 'South Ribble', 84: 'West Lancashire', 85: 'Wyre', 90: 'Knowsley', 91: 'Liverpool', 92: 'St. Helens', 93: 'Sefton', 95: 'Wirral', 100: 'Bolton', 101: 'Bury', 102: 'Manchester', 104: 'Oldham', 106: 'Rochdale', 
                       107: 'Salford', 109: 'Stockport', 110: 'Tameside', 112: 'Trafford', 114: 'Wigan', 120: 'Chester', 121: 'Congleton', 122: 'Crewe and Nantwich', 123: 'Ellesmere Port and Neston', 124: 'Halton', 126: 'Macclesfield',
                       127: 'Vale Royal', 128: 'Warrington', 129: 'Cheshire East', 130: 'Cheshire West and Chester', 139: 'Northumberland', 140: 'Alnwick', 141: 'Berwick-upon-Tweed', 142: 'Blyth Valley', 143: 'Castle Morpeth', 144: 'Tynedale',
                       145: 'Wansbeck', 146: 'Gateshead', 147: 'Newcastle upon Tyne', 148: 'North Tyneside', 149: 'South Tyneside', 150: 'Sunderland', 160: 'Chester-le-Street', 161: 'Darlington', 162: 'Derwentside', 163: 'Durham', 164: 'Easington', 
                       165: 'Sedgefield', 166: 'Teesdale', 168: 'Wear Valley', 169: 'County Durham', 180: 'Craven', 181: 'Hambleton', 182: 'Harrogate', 184: 'Richmondshire', 185: 'Ryedale', 186: 'Scarborough', 187: 'Selby', 189: 'York', 200: 'Bradford', 
                       202: 'Calderdale', 203: 'Kirklees', 204: 'Leeds', 206: 'Wakefield', 210: 'Barnsley', 211: 'Doncaster', 213: 'Rotherham', 215: 'Sheffield', 228: 'Kingston upon Hull, City of', 231: 'East Riding of Yorkshire', 232: 'North Lincolnshire', 
                       233: 'North East Lincolnshire', 240: 'Hartlepool', 241: 'Redcar and Cleveland', 243: 'Middlesbrough', 245: 'Stockton-on-Tees', 250: 'Cannock Chase', 251: 'East Staffordshire', 252: 'Lichfield', 253: 'Newcastle-under-Lyme', 254: 'South Staffordshire',
                       255: 'Stafford', 256: 'Staffordshire Moorlands', 257: 'Stoke-on-Trent', 258: 'Tamworth', 270: 'Bromsgrove', 273: 'Malvern Hills', 274: 'Redditch', 276: 'Worcester', 277: 'Wychavon', 278: 'Wyre Forest', 279: 'Bridgnorth', 280: 'North Shropshire', 281: 'Oswestry', 
                       282: 'Shrewsbury and Atcham', 283: 'South Shropshire', 284: 'Telford and Wrekin', 285: 'Herefordshire, County of ', 286: 'Shropshire', 290: 'North Warwickshire', 291: 'Nuneaton and Bedworth', 292: 'Rugby ', 293: 'Stratford-upon-Avon', 294: 'Warwick', 300: 'Birmingham',
                       302: 'Coventry', 303: 'Dudley', 305: 'Sandwell', 306: 'Solihull', 307: 'Walsall', 309: 'Wolverhampton', 320: 'Amber Valley', 321: 'Bolsover', 322: 'Chesterfield', 323: 'Derby', 324: 'Erewash', 325: 'High Peak', 327: 'North East Derbyshire', 328: 'South Derbyshire',
                       329: 'Derbyshire Dales', 340: 'Ashfield', 341: 'Bassetlaw', 342: 'Broxtowe', 343: 'Gedling', 344: 'Mansfield', 345: 'Newark and Sherwood', 346: 'Nottingham', 347: 'Rushcliffe', 350: 'Boston', 351: 'East Lindsey', 352: 'Lincoln', 353: 'North Kesteven', 
                       354: 'South Holland', 355: 'South Kesteven', 356: 'West Lindsey', 360: 'Blaby', 361: 'Hinckley and Bosworth', 362: 'Charnwood', 363: 'Harborough', 364: 'Leicester', 365: 'Melton', 366: 'North West Leicestershire', 367: 'Oadby and Wigston', 368: 'Rutland',
                       380: 'Corby', 381: 'Daventry', 382: 'East Northamptonshire', 383: 'Kettering', 384: 'Northampton', 385: 'South Northamptonshire', 386: 'Wellingborough', 390: 'Cambridge', 391: 'East Cambridgeshire', 392: 'Fenland', 393: 'Huntingdonshire', 394: 'Peterborough',
                       395: 'South Cambridgeshire', 400: 'Breckland', 401: 'Broadland', 402: 'Great Yarmouth', 404: 'Norwich', 405: 'North Norfolk', 406: 'South Norfolk', 407: "King's Lynn and West Norfolk", 410: 'Babergh', 411: 'Forest Heath', 412: 'Ipswich', 413: 'Mid Suffolk',
                       414: 'St. Edmundsbury', 415: 'Suffolk Coastal', 416: 'Waveney', 420: 'Bedford', 421: 'Luton', 422: 'Mid Bedfordshire', 423: 'South Bedfordshire', 424: 'Central Bedfordshire', 430: 'Broxbourne', 431: 'Dacorum', 432: 'East Hertfordshire', 433: 'North Hertfordshire', 
                       434: 'St. Albans', 435: 'Stevenage', 436: 'Three Rivers', 437: 'Watford', 438: 'Welwyn Hatfield', 450: 'Basildon', 451: 'Braintree', 452: 'Brentwood', 453: 'Castle Point', 454: 'Chelmsford', 455: 'Colchester', 456: 'Epping Forest', 457: 'Harlow', 458: 'Maldon', 
                       459: 'Rochford', 460: 'Southend-on-Sea', 461: 'Tendring', 462: 'Thurrock', 463: 'Uttlesford', 470: 'Bracknell Forest', 471: 'West Berkshire', 472: 'Reading', 473: 'Slough', 474: 'Windsor and Maidenhead', 475: 'Wokingham', 476: 'Aylesbury Vale', 477: 'South Bucks', 
                       478: 'Chiltern', 479: 'Milton Keynes', 480: 'Wycombe', 481: 'Cherwell', 482: 'Oxford', 483: 'Vale of White Horse', 484: 'South Oxfordshire', 485: 'West Oxfordshire', 490: 'Basingstoke and Deane', 491: 'Eastleigh', 492: 'Fareham', 493: 'Gosport', 
                       494: 'Hart', 495: 'Havant', 496: 'New Forest', 497: 'East Hampshire', 498: 'Portsmouth', 499: 'Rushmoor', 500: 'Southampton ', 501: 'Test Valley', 502: 'Winchester', 505: 'Isle of Wight', 510: 'Elmbridge', 511: 'Guildford', 512: 'Mole Valley', 513: 'Reigate and Banstead', 
                       514: 'Runnymede', 515: 'Surrey Heath', 516: 'Tandridge', 517: 'Waverley', 518: 'Woking', 530: 'Ashford', 531: 'Canterbury', 532: 'Dartford', 533: 'Dover', 535: 'Gravesham', 536: 'Maidstone', 538: 'Sevenoaks', 539: 'Shepway', 540: 'Swale', 541: 'Thanet', 542: 'Tonbridge and Malling', 
                       543: 'Tunbridge Wells', 544: 'Medway', 551: 'Eastbourne', 552: 'Hastings', 554: 'Lewes', 555: 'Rother', 556: 'Wealden', 557: 'Adur', 558: 'Arun', 559: 'Chichester', 560: 'Crawley', 562: 'Horsham', 563: 'Mid Sussex', 564: 'Worthing', 565: 'Brighton and Hove', 570: 'City of London', 
                       580: 'East Devon', 581: 'Exeter', 582: 'North Devon', 583: 'Plymouth', 584: 'South Hams', 585: 'Teignbridge', 586: 'Mid Devon', 587: 'Torbay', 588: 'Torridge', 589: 'West Devon', 590: 'Caradon', 591: 'Carrick', 592: 'Kerrier', 593: 'North Cornwall', 594: 'Penwith',
                       595: 'Restormel', 596: 'Cornwall', 601: 'Bristol, City of', 605: 'North Somerset', 606: 'Mendip', 607: 'Sedgemoor', 608: 'Taunton Deane', 609: 'West Somerset', 610: 'South Somerset', 611: 'Bath and North East Somerset', 612: 'South Gloucestershire', 620: 'Cheltenham', 
                       621: 'Cotswold', 622: 'Forest of Dean', 623: 'Gloucester', 624: 'Stroud', 625: 'Tewkesbury', 630: 'Kennet', 631: 'North Wiltshire', 632: 'Salisbury', 633: 'Swindon', 634: 'West Wiltshire', 635: 'Wiltshire', 640: 'Bournemouth', 641: 'Christchurch', 642: 'North Dorset', 643: 'Poole', 
                       644: 'Purbeck', 645: 'West Dorset', 646: 'Weymouth and Portland', 647: 'East Dorset', 720: 'Isle of Anglesey', 721: 'Conwy', 722: 'Gwynedd', 723: 'Denbighshire', 724: 'Flintshire', 725: 'Wrexham', 730: 'Blaenau Gwent', 731: 'Caerphilly', 732: 'Monmouthshire', 733: 'Newport', 
                       734: 'Torfaen', 740: 'Bridgend', 741: 'Cardiff', 742: 'Merthyr Tydfil', 743: 'Neath Port Talbot', 744: 'Rhondda, Cynon, Taff', 745: 'Swansea', 746: 'The Vale of Glamorgan', 750: 'Ceredigion', 751: 'Carmarthenshire', 752: 'Pembrokeshire', 753: 'Powys', 910: 'Aberdeen City', 
                       911: 'Aberdeenshire', 912: 'Angus', 913: 'Argyll and Bute', 914: 'Scottish Borders', 915: 'Clackmannanshire', 916: 'West Dunbartonshire', 917: 'Dumfries and Galloway', 918: 'Dundee City', 919: 'East Ayrshire', 920: 'East Dunbartonshire', 921: 'East Lothian', 922: 'East Renfrewshire', 
                       923: 'Edinburgh, City of', 924: 'Falkirk', 925: 'Fife', 926: 'Glasgow City', 927: 'Highland', 928: 'Inverclyde', 929: 'Midlothian', 930: 'Moray', 931: 'North Ayrshire', 932: 'North Lanarkshire', 933: 'Orkney Islands', 934: 'Perth and Kinross', 935: 'Renfrewshire', 936: 'Shetland Islands', 
                       937: 'South Ayrshire', 938: 'South Lanarkshire', 939: 'Stirling', 940: 'West Lothian', 941: 'Western Isles'}
        
        district = accidents['local_authority_district'].map(distr_guide)
        return district

    @staticmethod
    def assign_highway(accidents):
        higwy_guide = {'S12000033': 'Aberdeen City', 'S12000034': 'Aberdeenshire', 'S12000041': 'Angus', 'S12000035': 'Argyll & Bute', 'E09000002': 'Barking and Dagenham', 'E09000003': 'Barnet', 'E08000016': 'Barnsley', 'E06000022': 'Bath and North East Somerset', 'E06000055': 'Bedford', 'E09000004': 'Bexley',
                       'E08000025': 'Birmingham', 'E06000008': 'Blackburn with Darwen', 'E06000009': 'Blackpool', 'W06000019': 'Blaenau Gwent', 'E08000001': 'Bolton', 'E06000028': 'Bournemouth', 'E06000036': 'Bracknell Forest', 'E08000032': 'Bradford', 'E09000005': 'Brent', 'W06000013': 'Bridgend', 
                       'E06000043': 'Brighton and Hove', 'E06000023': 'Bristol, City of', 'E09000006': 'Bromley', 'E10000002': 'Buckinghamshire', 'E08000002': 'Bury', 'W06000018': 'Caerphilly', 'E08000033': 'Calderdale', 'E10000003': 'Cambridgeshire', 'E09000007': 'Camden', 'W06000015': 'Cardiff', 
                       'W06000010': 'Carmarthenshire', 'E06000056': 'Central Bedfordshire', 'W06000008': 'Ceredigion', 'E06000049': 'Cheshire East', 'E06000050': 'Cheshire West and Chester', 'E09000001': 'City of London', 'S12000005': 'Clackmannanshire', 'W06000003': 'Conwy', 'E06000052': 'Cornwall', 
                       'E06000047': 'County Durham', 'E08000026': 'Coventry', 'E09000008': 'Croydon', 'E10000006': 'Cumbria', 'E06000005': 'Darlington', 'W06000004': 'Denbighshire', 'E06000015': 'Derby', 'E10000007': 'Derbyshire', 'E10000008': 'Devon', 'E08000017': 'Doncaster', 'E10000009': 'Dorset', 
                       'E08000027': 'Dudley', 'S12000006': 'Dumfries & Galloway', 'S12000042': 'Dundee City', 'E09000009': 'Ealing', 'S12000008': 'East Ayrshire', 'S12000009': 'East Dunbartonshire', 'S12000010': 'East Lothian', 'S12000011': 'East Renfrewshire', 'E06000011': 'East Riding of Yorkshire', 
                       'E10000011': 'East Sussex', 'S12000036': 'Edinburgh, City of', 'E09000010': 'Enfield', 'E10000012': 'Essex', 'S12000014': 'Falkirk', 'S12000015': 'Fife', 'W06000005': 'Flintshire', 'E08000020': 'Gateshead', 'S12000043': 'Glasgow City', 'E10000013': 'Gloucestershire', 'E09000011': 'Greenwich', 
                       'W06000002': 'Gwynedd', 'E09000012': 'Hackney', 'E06000006': 'Halton', 'E09000013': 'Hammersmith and Fulham', 'E10000014': 'Hampshire', 'E09000014': 'Haringey', 'E09000015': 'Harrow', 'E06000001': 'Hartlepool', 'E09000016': 'Havering', 'E06000019': 'Herefordshire, County of ', 'E10000015': 'Hertfordshire',
                       'S12000017': 'Highland', 'E09000017': 'Hillingdon', 'E09000018': 'Hounslow', 'S12000018': 'Inverclyde', 'W06000001': 'Isle of Anglesey', 'E06000046': 'Isle of Wight', 'E06000053': 'Isles of Scilly', 'E09000019': 'Islington', 'E09000020': 'Kensington and Chelsea', 'E10000016': 'Kent',
                       'E06000010': 'Kingston upon Hull, City of', 'E09000021': 'Kingston upon Thames', 'E08000034': 'Kirklees', 'E08000011': 'Knowsley', 'E09000022': 'Lambeth', 'E10000017': 'Lancashire', 'E08000035': 'Leeds', 'E06000016': 'Leicester', 'E10000018': 'Leicestershire', 'E09000023': 'Lewisham', 
                       'E10000019': 'Lincolnshire', 'E08000012': 'Liverpool', 'EHEATHROW': 'London Airport (Heathrow)', 'E06000032': 'Luton', 'E08000003': 'Manchester', 'E06000035': 'Medway', 'W06000024': 'Merthyr Tydfil', 'E09000024': 'Merton', 'E06000002': 'Middlesbrough', 'S12000019': 'Midlothian', 
                       'E06000042': 'Milton Keynes', 'W06000021': 'Monmouthshire', 'S12000020': 'Moray', 'S12000013': 'Na h-Eileanan an Iar (Western Isles)', 'W06000012': 'Neath Port Talbot', 'E08000021': 'Newcastle upon Tyne', 'E09000025': 'Newham', 'W06000022': 'Newport', 'E10000020': 'Norfolk', 'S12000021': 'North Ayrshire',
                       'E06000012': 'North East Lincolnshire', 'S12000044': 'North Lanarkshire', 'E06000013': 'North Lincolnshire', 'E06000024': 'North Somerset', 'E08000022': 'North Tyneside', 'E10000023': 'North Yorkshire', 'E10000021': 'Northamptonshire', 'E06000048': 'Northumberland', 'E06000018': 'Nottingham', 
                       'E10000024': 'Nottinghamshire', 'E08000004': 'Oldham', 'S12000023': 'Orkney Islands', 'E10000025': 'Oxfordshire', 'W06000009': 'Pembrokeshire', 'S12000024': 'Perth and Kinross', 'E06000031': 'Peterborough', 'E06000026': 'Plymouth', 'E06000029': 'Poole', 'E06000044': 'Portsmouth', 'W06000023': 'Powys',
                       'E06000038': 'Reading', 'E09000026': 'Redbridge', 'E06000003': 'Redcar and Cleveland', 'S12000038': 'Renfrewshire', 'W06000016': 'Rhondda, Cynon, Taff', 'E09000027': 'Richmond upon Thames', 'E08000005': 'Rochdale', 'E08000018': 'Rotherham', 'E06000017': 'Rutland', 'E08000006': 'Salford', 'E08000028': 'Sandwell', 
                       'S12000026': 'Scottish Borders', 'E08000014': 'Sefton', 'E08000019': 'Sheffield', 'S12000027': 'Shetland Islands', 'E06000051': 'Shropshire', 'E06000039': 'Slough', 'E08000029': 'Solihull', 'E10000027': 'Somerset', 'S12000028': 'South Ayrshire', 'E06000025': 'South Gloucestershire', 
                       'S12000029': 'South Lanarkshire', 'E08000023': 'South Tyneside', 'E06000045': 'Southampton ', 'E06000033': 'Southend-on-Sea', 'E09000028': 'Southwark', 'E08000013': 'St. Helens', 'E10000028': 'Staffordshire', 'S12000030': 'Stirling', 'E08000007': 'Stockport', 'E06000004': 'Stockton-on-Tees', 'E06000021': 'Stoke-on-Trent',
                       'E10000029': 'Suffolk', 'E08000024': 'Sunderland', 'E10000030': 'Surrey', 'E09000029': 'Sutton', 'W06000011': 'Swansea', 'E06000030': 'Swindon', 'E08000008': 'Tameside', 'E06000020': 'Telford and Wrekin', 'W06000014': 'The Vale of Glamorgan', 'E06000034': 'Thurrock', 'E06000027': 'Torbay', 'W06000020': 'Torfaen', 
                       'E09000030': 'Tower Hamlets', 'E08000009': 'Trafford', 'E08000036': 'Wakefield', 'E08000030': 'Walsall', 'E09000031': 'Waltham Forest', 'E09000032': 'Wandsworth', 'E06000007': 'Warrington', 'E10000031': 'Warwickshire', 'E06000037': 'West Berkshire', 'S12000039': 'West Dunbartonshire', 'S12000040': 'West Lothian', 
                       'E10000032': 'West Sussex', 'E09000033': 'Westminster', 'E08000010': 'Wigan', 'E06000054': 'Wiltshire', 'E06000040': 'Windsor and Maidenhead', 'E08000015': 'Wirral', 'E06000041': 'Wokingham', 'E08000031': 'Wolverhampton', 'E10000034': 'Worcestershire', 'W06000006': 'Wrexham', 'E06000014': 'York'}
        
        highway = accidents['local_authority_highway'].map(higwy_guide)
        return highway
        
    @staticmethod
    def get_highway(accidents, longitude, latitude):
        """find and return the district of a given longitude&latitude coord"""
        
        cls = RoadAccidents()
        highway = cls.assign_highway(accidents)
        df = pd.concat([accidents[['longitude', 'latitude']], highway], axis=1)
        return df.loc[(df['longitude'] == longitude) &
                            (df['latitude'] == latitude)]
                            
    @staticmethod
    def get_district(accidents, longitude, latitude):
        """find and return the district of a given longitude&latitude coord"""
        
        cls = RoadAccidents()
        district = cls.assign_district(accidents)
        df = pd.concat([accidents[['longitude', 'latitude']], district], axis=1)
        return df.loc[(df['longitude'] == longitude) &
                            (df['latitude'] == latitude)]
       
    @staticmethod
    def assign_severity(accidents, col='casualty_severity'):
        guide = {1: 'fatal', 2: 'serious', 3: 'minor'}
        severity = accidents[col].map(guide)
        return severity
    
    @staticmethod
    def plot_accident_map(accidents, show_points_in_map:'subset of accidents', point_at='district',
                          main_color='gray', focus=False, focus_color='red', plot_title='UK 2019 Accidents',
                      title_size=20, figsize=(18, 6), alpha=None, dpi=250, savefig=False, fig_filename='accident_on_map.png'):
        """show accidents on scatterplot.
        accidents: main accidents df
        show_points_in_map: subset in accidents df
        point_at: {'district', 'highway'}
        main_color: background color
        focus: focus on the subset with same focus_color"""
        
        cls = RoadAccidents()
        
        
        condition = None
        if not focus:
            focus_color = focus_color
            if str.lower(point_at) == 'district':
                districts = cls.assign_district(accidents)
                condition = districts.loc[show_points_in_map.index]
            elif str.lower(point_at) == 'highway':
                highways = cls.assign_highway(accidents)
                condition = highways.loc[show_points_in_map.index]
        
        fig, axis = plt.subplots(figsize=figsize, dpi=dpi)
        sns.scatterplot(data=accidents, x='longitude', y='latitude', 
                       color=main_color, alpha=alpha, ax=axis)
        sns.scatterplot(data=show_points_in_map, x='longitude', y='latitude', hue=condition,
                        alpha=alpha, color=focus_color, ax=axis)
        
        axis.set_title(plot_title, size=title_size, weight='bold')
        
        if savefig:
            return print(cls.fig_writer(fig_filename, fig))
        return axis
        
    @staticmethod
    def get_accidents_when(accidents, col1, col1_is, col2=None, col2_is=None, col3=None, col3_is=None):
        """return filtered view of dataframe"""
        
        def show_with_one_cond(accidents, col1, col1_is):
            """return filtered view of dataframe"""
            if accidents is None:
                if isinstance(col1, pd.Series):
                    if isinstance(col1_is, (tuple, list)):
                        cond = (col1.isin(col1_is))
                    else:
                        cond = (col1 == col1_is)
                return col1.loc[cond]
            if isinstance(col1_is, (tuple, list)):
                cond = (accidents[col1].isin(col1_is))
            else:
                cond = (accidents[col1] == col1_is)
            return accidents.loc[cond]

        def show_with_two_cond(accidents, col1, col1_is, col2, col2_is):
            """return filtered view of dataframe"""
            
            result = show_with_one_cond(accidents, col1, col1_is)
            if isinstance(col2_is, (tuple, list)):
                cond = (result[col2].isin(col2_is))
            else:
                cond = (result[col2] == col2_is)
            return result.loc[cond]

        def show_with_three_cond(accidents, col1, col1_is, col2, col2_is, col3, col3_is):
            """return filtered view of dataframe"""
            
            result = show_with_two_cond(accidents, col1, col1_is, col2, col2_is)
            if isinstance(col3_is, (tuple, list)):
                cond = (result[col3].isin(col3_is))
            else:
                cond = (result[col3] == col3_is)
            return result.loc[cond]
        
        if col2 is not None and col2_is is not None:
            
            if col3 is not None and col3_is is not None:
                return show_with_three_cond(accidents, col1, col1_is, col2, col2_is, col3, col3_is)
            
            return show_with_two_cond(accidents, col1, col1_is, col2, col2_is)
        
        return show_with_one_cond(accidents, col1, col1_is)
        
    @staticmethod
    def show_weekly(accidents, week_num: int, check_for: str):
        """to display number of accidents for a given week_num"""
        
        selected_wk = accidents.loc[accidents['week_num'].isin([week_num]), check_for]
        # weekday
        unqs = accidents[check_for].unique()
        for unq in unqs:
            print(f"For week {week_num}'s {check_for} accidents:\n" +
                  f'{unq} = {selected_wk.loc[selected_wk == unq].shape[0]}')
        
    @staticmethod
    def quarterly_observations(accidents, suffix, include_perc=False):
            """visualize quarterly observations"""
            
            cls = RoadAccidents()
            agg_df = cls.generate_aggregated_lookup(accidents)
            fig = plt.figure(figsize=(30, 18), dpi=200)
            bl, br = fig.add_axes([0, 0, 0.6, 0.6]), fig.add_axes([0.68, 0.1, 0.35, 0.4])
            tl, tr = fig.add_axes([0, 0.65, 0.6, 0.6]), fig.add_axes([0.65, 0.75, 0.35, 0.4])
            # QuarterLY OBSERVATIONS
            # weekend accidents
            x, z, y = ['quarter', 'is_weekend', 'total_count']
            acc_mon_wkend = agg_df[[x, z, y]].groupby([x, z]).sum().sort_index().reset_index()
            perc = np.round(100*acc_mon_wkend['total_count']/acc_mon_wkend['total_count'].sum(),
                            2)
            cls.plot_column(acc_mon_wkend[x], acc_mon_wkend[y], condition_on=acc_mon_wkend[z],
                             plot_title='Weekend Vs Weekday Accidents per Quarter', annotate=True, annot_size=12,
                             include_perc=include_perc,  top_labe_gap=1500, h_labe_shift=-0.15,  xy_labe_size=18,
                             axis=tl)

            # seasonal accidents
            x, b, z, y = ['quarter', 'season_num', 'season', 'total_count']
            acc_mon_ssn = agg_df[[x, b, z, y]].groupby([x, z]).sum().sort_index().reset_index()
            perc = np.round(100*acc_mon_ssn['total_count']/acc_mon_ssn['total_count'].sum(),
                            2)

            cls.plot_column(acc_mon_ssn[x], acc_mon_ssn[y], condition_on=acc_mon_ssn[z],
                             plot_title='Seasonal Accidents per Quarter', annotate=True, annot_size=10,
                            paletter={'autumn': 'black', 'spring': 'green', 'summer': 'yellow', 'winter':'gray'},
                             include_perc=include_perc,  top_labe_gap=1500, h_labe_shift=-0.15,
                              xy_labe_size=14, axis=tr)
            sns.move_legend(tr, [0.75, 0.7])

            # day of week accidents
            x, b, z, y = ['quarter', 'day_of_week', 'day_name', 'total_count']
            acc_mon_dow = agg_df[[x, b, z, y]].groupby([x, b, z]).sum().sort_index().reset_index()
            perc = np.round(100*acc_mon_dow['total_count']/acc_mon_dow['total_count'].sum(),
                            2)

            color_mapping = {'Sunday': 'green', 'Monday': 'blue', 'Tuesday': 'yellow', 'Wednesday': 'darkorange', 'Thursday': 'red', 'Friday': 'black', 'Saturday': 'gray'}

            cls.plot_column(acc_mon_dow[x], acc_mon_dow[y], condition_on=acc_mon_dow[z],
                             plot_title='Day of Week Accidents per Quarter', annotate=True, annot_size=12,
                            paletter=color_mapping, include_perc=include_perc,  top_labe_gap=400, 
                              xy_labe_size=14, h_labe_shift=-0.015, axis=bl)
            #bl.set_ylim(top=15000)
            sns.move_legend(bl, loc=[1.005, 0.5])


            # active hour accidents
            x, z, y = ['quarter', 'inactive_hour', 'total_count']
            acc_mon_aut = agg_df[[x, z, y]].groupby([x, z]).sum().sort_index().reset_index()
            perc = np.round(100*acc_mon_aut['total_count']/acc_mon_aut['total_count'].sum(),
                            2)

            cls.plot_column(acc_mon_aut[x], acc_mon_aut[y], condition_on=acc_mon_aut[z], annot_size=10,
                             plot_title='Active Vs. Inactive Hour Accidents per Quarter', annotate=True,
                             paletter={0:'lightgreen', 1:'darkblue'}, xy_labe_size=14,
                             include_perc=include_perc,  top_labe_gap=2500, h_labe_shift=-0.15, axis=br)
            
            #br.set_ylim(top=70000)
            sns.move_legend(bl, loc=[1.005, 0.5])
            fig_filename = f'quarterly_{suffix}.png'
            print(cls.fig_writer(fig_filename, fig))
        
    @staticmethod
    def monthly_observations(accidents, suffix, include_perc=False):
            """visualize monthly observations"""
            
            cls = RoadAccidents()
            agg_df = cls.generate_aggregated_lookup(accidents)
            fig = plt.figure(figsize=(30, 18), dpi=200)
            bl, br = fig.add_axes([0, 0, 0.6, 0.6]), fig.add_axes([0.68, 0.1, 0.35, 0.4])
            tl, tr = fig.add_axes([0, 0.65, 0.6, 0.6]), fig.add_axes([0.65, 0.75, 0.35, 0.4])
            # MonthLY OBSERVATIONS
            # weekend accidents
            x, z, y = ['month', 'is_weekend', 'total_count']
            acc_mon_wkend = agg_df[[x, z, y]].groupby([x, z]).sum().reset_index()
            perc = np.round(100*acc_mon_wkend['total_count']/acc_mon_wkend['total_count'].sum(),
                            2)
            cls.plot_column(acc_mon_wkend[x], acc_mon_wkend[y], condition_on=acc_mon_wkend[z],
                             plot_title='Weekend Vs Weekday Accidents per Month', annotate=True, annot_size=14,
                             include_perc=include_perc,  top_labe_gap=500, h_labe_shift=-0.05, axis=tl)

            # seasonal accidents
            x, z, y = ['month', 'season', 'total_count']
            acc_mon_aut = agg_df[[x, z, y]].groupby([x, z]).sum().reset_index()
            perc = np.round(100*acc_mon_aut['total_count']/acc_mon_aut['total_count'].sum(),
                            2)

            cls.plot_column(acc_mon_aut[x], acc_mon_aut[y], condition_on=acc_mon_aut[z],
                             plot_title='Seasonal Accidents per Month', annotate=True, annot_size=10,
                            paletter={'autumn': 'black', 'spring': 'green', 'summer': 'yellow', 'winter':'gray'},
                             axis=tr)
            sns.move_legend(tr, [0.42, 0.7])

            # day of week accidents
            x, b, z, y = ['month', 'day_of_week', 'day_name', 'total_count']
            acc_mon_aut = agg_df[[x, b, z, y]].groupby([x, b, z]).sum().sort_index().reset_index()
            perc = np.round(100*acc_mon_aut['total_count']/acc_mon_aut['total_count'].sum(),
                            2)

            color_mapping = {'Sunday': 'green', 'Monday': 'blue', 'Tuesday': 'yellow', 'Wednesday': 'darkorange',
                            'Thursday': 'red', 'Friday': 'black', 'Saturday': 'gray'}

            cls.plot_column(acc_mon_aut[x], acc_mon_aut[y], condition_on=acc_mon_aut[z],
                             plot_title='Day of Week Accidents per Month', annotate=True, annot_size=11,
                            paletter=color_mapping, include_perc=include_perc,  top_labe_gap=100, 
                             h_labe_shift=0.05,axis=bl)
            sns.move_legend(bl, loc=[1.005, 0.5])


            # active hour accidents
            x, z, y = ['month', 'inactive_hour', 'total_count']
            acc_mon_aut = agg_df[[x, z, y]].groupby([x, z]).sum().sort_index().reset_index()
            perc = np.round(100*acc_mon_aut['total_count']/acc_mon_aut['total_count'].sum(),
                            2)

            cls.plot_column(acc_mon_aut[x], acc_mon_aut[y], condition_on=acc_mon_aut[z],
                             plot_title='Active Vs. Inactive Hour Accidents per Month', annotate=True,
                             paletter={0:'lightgreen', 1:'darkblue'}, include_perc=include_perc, 
                             annot_size=10, top_labe_color='red', top_labe_gap=-700, h_labe_shift=-0.05, axis=br)
            
            fig_filename = f'monthly_{suffix}.png'
            print(cls.fig_writer(fig_filename, fig))
        
    @staticmethod
    def weekly_observations(accidents, suffix):
        """visualize weekly observations"""
        
        cls = RoadAccidents()
        agg_df = cls.generate_aggregated_lookup(accidents)
        fig = plt.figure(figsize=(30, 18), dpi=200)
        bl, br = fig.add_axes([0, 0, 0.6, 0.6]), fig.add_axes([0.64, 0.1, 0.35, 0.4])
        tl, tr = fig.add_axes([0, 0.65, 0.63, 0.6]), fig.add_axes([0.65, 0.75, 0.35, 0.4])
        
        # WeekLY OBSERVATIONS
        # weekend accidents
        x, z, y = ['week_num', 'is_weekend', 'total_count']
        acc_mon_wkend = agg_df[[x, z, y]].groupby([x, z]).sum().reset_index()
        
        cls.plot_column(acc_mon_wkend[x], acc_mon_wkend[y], condition_on=acc_mon_wkend[z],
                         plot_title='Weekend Vs Weekday Accidents per Week', annotate=True, annot_size=13,
                         top_labe_gap=1500, h_labe_shift=-0.15, axis=tl)

        # seasonal accidents
        x, z, y = ['week_num', 'season', 'total_count']
        acc_mon_aut = agg_df[[x, z, y]].groupby([x, z]).sum().reset_index()
        acc_mon_aut

        cls.plot_scatter(acc_mon_aut[x], acc_mon_aut[y], condition_on=acc_mon_aut[z],
                         plot_title='Seasonal Accidents per Week', 
                        paletter={'autumn': 'black', 'spring': 'green', 'summer': 'yellow', 'winter':'gray'},
                         axis=tr)
        sns.move_legend(tr, [0.55, 0.2])
        
        # active hour accidents
        x, z, y = ['week_num', 'inactive_hour', 'total_count']
        acc_mon_aut = agg_df[[x, z, y]].groupby([x, z]).sum().sort_index().reset_index()
        acc_mon_aut

        cls.plot_column(acc_mon_aut[x], acc_mon_aut[y], condition_on=acc_mon_aut[z],
                         plot_title='Active Vs. Inactive Hour Accidents per Week', 
                         paletter={0:'lightgreen', 1:'darkblue'}, annot_size=14,
                         axis=bl)

        # day of week accidents
        x, b, z, y = ['week_num', 'day_of_week', 'day_name', 'total_count']
        acc_mon_aut = agg_df[[x, b, z, y]].groupby([x, b, z]).sum().sort_index().reset_index()
        acc_mon_aut

        color_mapping = {'Sunday': 'green', 'Monday': 'blue', 'Tuesday': 'yellow', 'Wednesday': 'darkorange',
                        'Thursday': 'red', 'Friday': 'black', 'Saturday': 'gray'}

        cls.plot_scatter(acc_mon_aut[x], acc_mon_aut[y], condition_on=acc_mon_aut[z],
                         plot_title='Day of Week Accidents per Week', 
                        paletter=color_mapping, axis=br)
        
        sns.move_legend(br, loc=[0.05, 0.75])
        
        fig_filename = f'weekly_{suffix}.png'
        print(cls.fig_writer(fig_filename, fig))
        
    @staticmethod
    def daily_observations(accidents, suffix):
        """visualize daily observations"""
        
        cls = RoadAccidents()
        agg_df = cls.generate_aggregated_lookup(accidents)
        fig, ((tl, tr), (bl, br)) = plt.subplots(2, 2, figsize=(30, 18), dpi=200)
        # Daily OBSERVATIONS
        # hour accidents
        x, z, y = ['day_num', 'is_weekend', 'total_count']
        acc_mon_wkend = agg_df[[x, z, y]].groupby([x, z]).sum().reset_index()
        acc_mon_wkend
        cls.plot_scatter(acc_mon_wkend[x], acc_mon_wkend[y], condition_on=acc_mon_wkend[z],
                         plot_title='Weekend Vs Weekday Accidents per Day', 
                         axis=tl)

        # seasonal accidents
        x, z, y = ['day_num', 'season', 'total_count']
        acc_mon_aut = agg_df[[x, z, y]].groupby([x, z]).sum().reset_index()
        acc_mon_aut

        cls.plot_scatter(acc_mon_aut[x], acc_mon_aut[y], condition_on=acc_mon_aut[z],
                         plot_title='Seasonal Accidents per Day', 
                        paletter={'autumn': 'black', 'spring': 'green', 'summer': 'yellow', 'winter':'gray'},
                         axis=tr)
        tr.set_ylim(bottom=250)
        sns.move_legend(tr, [0.46, 0.10])

        # day of hour accidents
        x, b, z, y = ['day_num', 'day_of_week', 'day_name', 'total_count']
        acc_mon_aut = agg_df[[x, b, z, y]].groupby([x, b, z]).sum().sort_index().reset_index()
        acc_mon_aut

        color_mapping = {'Sunday': 'green', 'Monday': 'blue', 'Tuesday': 'yellow', 'Wednesday': 'darkorange', 
                        'Thursday': 'red', 'Friday': 'black', 'Saturday': 'gray'}

        cls.plot_scatter(acc_mon_aut[x], acc_mon_aut[y], condition_on=acc_mon_aut[z],
                         plot_title='Day of Year Accidents per Day', 
                        paletter=color_mapping, axis=bl)
        sns.move_legend(bl, loc=[1.005, 0.5])


        # active hour accidents
        x, z, y = ['day_num', 'inactive_hour', 'total_count']
        acc_mon_aut = agg_df[[x, z, y]].groupby([x, z]).sum().sort_index().reset_index()
        acc_mon_aut

        cls.plot_scatter(acc_mon_aut[x], acc_mon_aut[y], condition_on=acc_mon_aut[z],
                         plot_title='Active Vs. Inactive Hour Accidents per Day', 
                         paletter={0:'lightgreen', 1:'darkblue'},
                         axis=br)
        
        fig_filename = f'daily_{suffix}.png'
        print(cls.fig_writer(fig_filename, fig))
    
    @staticmethod
    def hourly_observations(accidents, suffix):
            """visualize hourly observations"""
            
            cls = RoadAccidents()
            agg_df = cls.generate_aggregated_lookup(accidents)
            
            fig = plt.figure(figsize=(30, 18), dpi=200)
            bl, br = fig.add_axes([0, 0, 0.65, 0.6]), fig.add_axes([0.68, 0.1, 0.35, 0.4])
            tl, tr = fig.add_axes([0, 0.65, 0.6, 0.6]), fig.add_axes([0.65, 0.75, 0.35, 0.4])
            # HourLY OBSERVATIONS
            # seasonal accidents
            x, z, y = ['hour', 'is_weekend', 'total_count']
            acc_mon_wkend = agg_df[[x, z, y]].groupby([x, z]).sum().reset_index()
            acc_mon_wkend
            cls.plot_column(acc_mon_wkend[x], acc_mon_wkend[y], condition_on=acc_mon_wkend[z],
                             plot_title='Weekend Vs Weekday Accidents per Hour', annotate=True,
                             annot_size=16, axis=tl)

            # seasonal accidents
            x, z, y = ['hour', 'season', 'total_count']
            acc_mon_aut = agg_df[[x, z, y]].groupby([x, z]).sum().reset_index()
            acc_mon_aut

            cls.plot_column(acc_mon_aut[x], acc_mon_aut[y], condition_on=acc_mon_aut[z], annotate=False,
                             plot_title='Seasonal Accidents per Hour', annot_size=10,
                            paletter={'autumn': 'black', 'spring': 'green', 'summer': 'yellow', 'winter':'gray'},
                             axis=tr)
            sns.move_legend(tr, [0.42, 0.7])

            # day of week accidents
            x, b, z, y = ['hour', 'day_of_week', 'day_name', 'total_count']
            acc_mon_aut = agg_df[[x, b, z, y]].groupby([x, b, z]).sum().sort_index().reset_index()
            acc_mon_aut

            color_mapping = {'Sunday': 'green', 'Monday': 'blue', 'Tuesday': 'yellow', 'Wednesday': 'darkorange',
                            'Thursday': 'red', 'Friday': 'black', 'Saturday': 'gray'}

            cls.plot_column(acc_mon_aut[x], acc_mon_aut[y], condition_on=acc_mon_aut[z],
                             plot_title='Day of Week Accidents per Hour', annotate=False,
                            annot_size=12, paletter=color_mapping, axis=bl)


            # active hour accidents
            x, z, y = ['hour', 'inactive_hour', 'total_count']
            acc_mon_aut = agg_df[[x, z, y]].groupby([x, z]).sum().sort_index().reset_index()
            acc_mon_aut

            cls.plot_column(acc_mon_aut[x], acc_mon_aut[y], condition_on=acc_mon_aut[z],
                             plot_title='Active Vs. Inactive Hour Accidents per Hour', annotate=True,
                             paletter={0:'lightgreen', 1:'darkblue'}, annot_size=12.5,
                             axis=br)
            
            fig_filename = f'hourly_{suffix}.png'
            print(cls.fig_writer(fig_filename, fig))
    
    @staticmethod
    def seasonal_observations(accidents, suffix='overall', include_perc=False):
    
        cls = RoadAccidents()
        agg_df = cls.generate_aggregated_lookup(accidents)
        # seasonal observations
        cols = ['season_num', 'season', 'inactive_hour', 'day_of_week', 'day_name', 'is_weekend',
                'part_of_day_num', 'part_of_day', 'total_count']
        total_ssn = agg_df[cols].groupby(cols[:-1]).sum().sort_index().reset_index()
        # display(total_ssn)
        
        # season per inactive_hour
        cols = ['season_num', 'season', 'inactive_hour', 'total_count']
        total_ssn_inacv = total_ssn[cols].groupby(cols[:-1]).sum().sort_index().reset_index()

        color_mapping = {'autumn': 'black', 'spring': 'green', 'summer': 'yellow', 'winter':'gray'},

        fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
        # avg line
        avg = total_ssn_inacv['total_count'].mean().round(2)
        cls.plot_line([-1, len(total_ssn)+1], [avg, avg], color='black', axis=ax)
        labe = f'Average: {avg}'
        ax.text(1.05, avg, labe, color='white',size=7, bbox={'edgecolor':'none', 'facecolor':'black'})

        # column
        cls.plot_column(total_ssn_inacv['season'], total_ssn_inacv['total_count'], total_ssn_inacv['inactive_hour'], 
                       paletter={1: 'gray', 0:'red'}, annot_size=14, y_labe='total_count', x_labe='season',
                        xy_labe_size=18, h_labe_shift=-0.05, top_labe_gap=-5000, include_perc=include_perc,
                        top_labe_color='lightgreen',
                         plot_title='Seasonal Accidents per Hourly Activity', axis=ax)
        sns.move_legend(ax, loc=[1.01, 0.65])
        fname = f'hist_ssn_inactv_{suffix}.png'
        plt.show()
        print(cls.fig_writer(fname, fig))


        # season accidents per part of day
        cols = ['season_num', 'season', 'part_of_day_num','part_of_day', 'total_count']
        total_ssn_pod = total_ssn[cols].groupby(cols[:-1]).sum().sort_index().reset_index()

        fig, ax = plt.subplots(figsize=(12, 7), dpi=200)
        cls.plot_column(total_ssn_pod['season'], total_ssn_pod['total_count'], total_ssn_pod['part_of_day'], 
                         paletter={'morning':'orange', 'afternoon':'yellow', 'evening':'gray', 'night':'black'}, 
                         plot_title='Seasonal Accidents per Part of the Day', annot_size=11, y_labe='total_count',  
                         x_labe='season', xy_labe_size=14, include_perc=include_perc, top_labe_gap=-1800,
                         top_labe_color='darkgreen',
                        h_labe_shift=-0.01, axis=ax)
        sns.move_legend(ax, loc=[1.01, 0.65])
        plt.show()
        fname = f'ssn_accidents_pod_{suffix}.png'
        print(cls.fig_writer(fname, fig))
        
        # season accidents per part of day
        cols = ['season_num', 'season', 'day_of_week', 'day_name', 'total_count']
        total_ssn_wkend = total_ssn[cols].groupby(cols[:-1]).sum().sort_index().reset_index()

        fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
        color_mapping = {'Sunday': 'green', 'Monday': 'blue', 'Tuesday': 'yellow', 'Wednesday': 'darkorange',
                         'Thursday': 'red', 'Friday': 'black', 'Saturday': 'gray'}
        cls.plot_column(total_ssn_wkend['season'], total_ssn_wkend['total_count'], total_ssn_wkend['day_name'], 
                         paletter=color_mapping, 
                         plot_title='Seasonal Accidents per Weekendly Activity',
                         annot_size=7.5, y_labe='total_count', x_labe='season', xy_labe_size=14, axis=ax)
        sns.move_legend(ax, loc=[1.01, 0.65])
        plt.show()
        fname = f'ssn_accidents_dayname_{suffix}.png'
        print(cls.fig_writer(fname, fig))


        # season accidents per part of day
        cols = ['season_num', 'season', 'is_weekend', 'total_count']
        total_ssn_wkend = total_ssn[cols].groupby(cols[:-1]).sum().sort_index().reset_index()

        fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
        avg = total_ssn_wkend['total_count'].mean().round(2)
        cls.plot_line([-1, len(total_ssn)+1], [avg, avg], color='black', axis=ax)
        labe = f'Average: {avg}'
        ax.text(1.05, avg, labe, color='white',size=7, bbox={'edgecolor':'none', 'facecolor':'black'})

        cls.plot_column(total_ssn_wkend['season'], total_ssn_wkend['total_count'], total_ssn_wkend['is_weekend'], 
                         paletter={0:'blue', 1:'green'}, 
                         plot_title='Seasonal Accidents per Weekendly Activity',
                         annot_size=11, y_labe='total_count', x_labe='season', xy_labe_size=14, include_perc=include_perc,
                        h_labe_shift=-0.05, top_labe_gap=-4000, top_labe_color='white', axis=ax)

        plt.show()
        fname = f'ssn_accidents_wkend_{suffix}.png'
        print(cls.fig_writer(fname, fig))
    
    def dst_observations(accidents, include_perc=False, suffix='ovr'):
        
        cls = RoadAccidents()
        total_dst = cls.generate_aggregated_lookup(accidents)
        # dst accidents per ssn
        cols = ['is_dst', 'season_num', 'season', 'total_count']
        total_dst_ssn = total_dst[cols].groupby(cols[:-1]).sum().sort_index().reset_index()

        fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
        color_mapping = {'autumn': 'black', 'spring': 'green', 'summer': 'yellow', 'winter':'gray'}
        cls.plot_column(total_dst_ssn['is_dst'], total_dst_ssn['total_count'], total_dst_ssn['season'], 
                         paletter=color_mapping, include_perc=include_perc, top_labe_gap=-3000, h_labe_shift=-0.05,
                         plot_title='DST-sensitive Accidents per Season', top_labe_color='white',
                         annot_size=12, y_labe='total_count', x_labe='DST', xy_labe_size=14, axis=ax)
        sns.move_legend(ax, loc=[1.01, 0.65])
        plt.show()
        fname = f'dst_accidents_ssn_{suffix}.png'
        print(cls.fig_writer(fname, fig))
        
        
        # dst accidents per quarter
        cols = ['is_dst', 'quarter', 'total_count']
        total_dst_q = total_dst[cols].groupby(cols[:-1]).sum().sort_index().reset_index()

        fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
        color_mapping = {1:'orange', 2:'green', 3:'yellow', 4:'red'}
        cls.plot_column(total_dst_q['is_dst'], total_dst_q['total_count'], total_dst_q['quarter'], 
                         paletter=color_mapping, include_perc=include_perc, top_labe_gap=-3000, h_labe_shift=-0.05,
                         plot_title='DST-sensitive Accidents per Quarter', top_labe_color='white',
                         annot_size=12, y_labe='total_count', x_labe='DST', xy_labe_size=14, axis=ax)
        sns.move_legend(ax, loc=[1.01, 0.65])
        plt.show()
        fname = f'dst_accidents_q_{suffix}.png'
        print(cls.fig_writer(fname, fig))
        

        # dst accidents per month
        cols = ['month', 'is_dst', 'month_name', 'total_count']
        total_dst_q = total_dst[cols].groupby(cols[:-1]).sum().sort_index().reset_index()

        fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
        color_mapping = {0:'gray', 1:'green'}
        cls.plot_column(total_dst_q['month_name'], total_dst_q['total_count'], total_dst_q['is_dst'], 
                         include_perc=False, top_labe_gap=-3000, h_labe_shift=-0.05, paletter=color_mapping,
                         plot_title='DST-sensitive Accidents per month_name', top_labe_color='black',
                         annot_size=12, y_labe='total_count', x_labe='Month', xy_labe_size=14, axis=ax)
        sns.move_legend(ax, loc=[1.01, 0.65])
        plt.show()
        fname = f'dst_accidents_mn_{suffix}.png'
        print(cls.fig_writer(fname, fig))
        
        
        # dst accidents per week_num
        cols = ['week_num', 'is_dst', 'total_count']
        total_dst_wk = total_dst[cols].groupby(cols[:-1]).sum().sort_index().reset_index()

        fig, ax = plt.subplots(figsize=(22, 8), dpi=200)
        color_mapping = {0:'gray', 1:'green'}
        cls.plot_column(total_dst_wk['week_num'], total_dst_wk['total_count'], total_dst_wk['is_dst'], 
                         include_perc=False, top_labe_gap=-3000, h_labe_shift=-0.05, paletter=color_mapping,
                         plot_title='DST-sensitive Accidents per Week of Year', top_labe_color='black',
                         annot_size=10, y_labe='total_count', x_labe='Week Number', xy_labe_size=14, axis=ax)
        sns.move_legend(ax, loc=[1.01, 0.65])
        plt.show()
        fname = f'dst_accidents_wk_{suffix}.png'
        print(cls.fig_writer(fname, fig))
        
        
        # dst accidents per hour
        cols = ['hour', 'is_dst', 'total_count']
        total_dst_hr = total_dst[cols].groupby(cols[:-1]).sum().sort_index().reset_index()

        fig, ax = plt.subplots(figsize=(24, 6), dpi=200)
        color_mapping = {0:'gray', 1:'green'}
        cls.plot_column(total_dst_hr['hour'], total_dst_hr['total_count'], total_dst_hr['is_dst'], 
                         include_perc=False, top_labe_gap=-3000, h_labe_shift=-0.05, paletter=color_mapping,
                         plot_title='DST-sensitive Accidents per Hour of Day', top_labe_color='black',
                         annot_size=12.5, y_labe='total_count', x_labe='Hour', xy_labe_size=14, axis=ax)
        sns.move_legend(ax, loc=[1.01, 0.65])
        plt.show()
        fname = f'dst_accidents_hr_{suffix}.png'
        print(cls.fig_writer(fname, fig))
        
        
        # dst accidents per day_name
        cols = ['day_of_week', 'day_name', 'is_dst', 'total_count']
        total_dst_dayname = total_dst[cols].groupby(cols[:-1]).sum().sort_index().reset_index()

        fig, ax = plt.subplots(figsize=(15, 6), dpi=200)
        color_mapping = {0:'gray', 1:'green'}
        cls.plot_column(total_dst_dayname['day_name'], total_dst_dayname['total_count'], total_dst_dayname['is_dst'], 
                         include_perc=False, top_labe_gap=-3000, h_labe_shift=-0.05, paletter=color_mapping,
                         plot_title='DST-sensitive Accidents per Day of Week', top_labe_color='black',
                         annot_size=12.5, y_labe='total_count', x_labe='day_name', xy_labe_size=14, axis=ax)
        sns.move_legend(ax, loc=[1.01, 0.65])
        plt.show()
        fname = f'dst_accidents_dayname_{suffix}.png'
        print(cls.fig_writer(fname, fig))
        
        
        # dst accidents per part_of_day
        cols = ['part_of_day_num', 'part_of_day', 'is_dst', 'total_count']
        total_dst_pod = total_dst[cols].groupby(cols[:-1]).sum().sort_index().reset_index()

        fig, ax = plt.subplots(figsize=(15, 6), dpi=200)
        color_mapping = {0:'gray', 1:'green'}
        cls.plot_column(total_dst_pod['part_of_day'], total_dst_pod['total_count'], total_dst_pod['is_dst'], 
                         include_perc=False, top_labe_gap=-3000, h_labe_shift=-0.05, paletter=color_mapping,
                         plot_title='DST-sensitive Accidents for Parts of Day', top_labe_color='black',
                         annot_size=12.5, y_labe='total_count', x_labe='part_of_day', xy_labe_size=14, axis=ax)
        sns.move_legend(ax, loc=[1.01, 0.65])
        plt.show()
        fname = f'dst_accidents_pod_{suffix}.png'
        print(cls.fig_writer(fname, fig))
        
        
        # dst accidents per is_weekend
        cols = ['is_dst', 'is_weekend', 'total_count']
        total_dst_wkend = total_dst[cols].groupby(cols[:-1]).sum().sort_index().reset_index()

        fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
        color_mapping = {0:'gray', 1:'green'}
        cls.plot_column(total_dst_wkend['is_weekend'], total_dst_wkend['total_count'], total_dst_wkend['is_dst'], 
                         include_perc=False, top_labe_gap=-3000, h_labe_shift=-0.05, paletter=color_mapping,
                         plot_title='DST-sensitive Accidents for Weekday Vs Weekend', top_labe_color='black',
                         annot_size=12.5, y_labe='total_count', x_labe='is_weekend', xy_labe_size=14, axis=ax)
        sns.move_legend(ax, loc=[1.01, 0.65])
        plt.show()
        fname = f'dst_accidents_wkend_{suffix}.png'
        print(cls.fig_writer(fname, fig))
    
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
        rng = np.random.default_rng
        
        tstat, pval = np.round(stats.ttest_ind(x1, x2, equal_var=False, alternative=h1, random_state=rng), 4)
        return tstat, pval
        
    @staticmethod
    def assign_seasons(df, in_words=True):
        """generate 2019 UK seasons from month and day columns using.
        1. winter: december 21 - march 19
        2. spring: march 20 - june 20
        3. summer: june 21 - september 22
        4. autumn: september 23 - december 20
        
        steps
        1. select full months
        2. exclude outside days
        Return:
        seasons: series in words or integer"""
       
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
        if in_words:
            result.loc[spring_ind] = 'spring'
            result.loc[summer_ind] = 'summer'
            result.loc[autumn_ind] = 'autumn'
            result.loc[winter_ind] = 'winter'
        else:
            result.loc[winter_ind] = 1
            result.loc[spring_ind] = 2
            result.loc[summer_ind] = 3
            result.loc[autumn_ind] = 4
        return result
        
    @staticmethod
    def compute_stat_pval(x1, x2, h1='greater',  X1_name='X1', X2_name='X2', deg_freedom=1, seed=1):
        """report statistical significance using pvalue"""
    
        def get_min_denom(n1, n2):
                return min([n1, n2])
            
        def detect_unequal_sizes(n1, n2):
            """check if both lengths are not the same"""
            return n1 != n2
        
        cls = RoadAccidents()

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
        
        tstat, pval = cls.compute_pvalue(samp_sizes[X1_name], samp_sizes[X2_name], h1, seed)
        test_result = (['99%', 0.01], 
                       ['95%', 0.05])
        if pval <= test_result[0][1]:
            return print(f'At {test_result[0][0]}, REJECT the null hypothesis!\n {pval} is less than {test_result[0][1]}\n')
        elif pval <= test_result[1][1]:
                return print(f'At {test_result[1][0]}, REJECT the null hypothesis!\n{pval} is less than or equal to {test_result[1][1]}\n')
        print(f'Do NOT reject the null hypothesis\n{pval} is greater than {test_result[1][1]}')
        
    @staticmethod
    def hourly_trend(database, fname_suffix='overall'):
    
        cls = RoadAccidents()
        general_agg = cls.generate_aggregated_lookup(database)
        # 1. average hourly accident per day of week showing weekend and weekdays
        cols = ['hour', 'day_name', 'is_weekend', 'total_count']
        acc_hr_wkend = general_agg[cols].groupby(cols[:-1]).mean().reset_index()
        #print(acc_hr_wkend)

        color_mapping = {0: 'red', 1: 'gray'}

        fig, (l,r) = plt.subplots(1, 2, figsize=(28, 14), dpi=150)
        cls.plot_column(acc_hr_wkend['hour'], acc_hr_wkend['total_count'], annotate=False, condition_on=acc_hr_wkend['is_weekend'],
                        paletter=color_mapping, plot_title=f'Weekend Vs. Weekday Average Accidents per Hour',
                        axis=l)
        line1 = "From 22pm to 4am and 10am to 13pm, there were more accidents on weekends.\n"
        line2 = "From 5am to 9am and 14pm to 19pm, there were more accidents on weekdays."
        annotation = line1+line2
        l.text(0, 4.8, annotation, color='black', weight='bold', size=13,
               bbox={'facecolor': 'none', 'edgecolor':'blue'})
        l.set_ylim(top=5.5)

        # 2. total accidents per day of week
        cols = ['hour', 'day_of_week', 'day_name', 'total_count']
        acc_hr_dayname = general_agg[cols].groupby(cols[:-1]).sum().reset_index()
        acc_hr_dayname

        color_mapping = {'Sunday': 'green', 'Monday': 'blue', 'Tuesday': 'yellow', 'Wednesday': 'darkorange', 'Thursday': 'red', 'Friday': 'black', 'Saturday': 'gray'}
        cls.plot_column(acc_hr_dayname['hour'], acc_hr_dayname['total_count'], annotate=False, 
                        condition_on=acc_hr_dayname['day_name'], paletter=color_mapping, axis=r, 
                        plot_title='Total Hourly Accidents per Day of Week')
        line1 = "From 0am to 4am, highest accidents were on Saturdays & Sundays.\n"
        line2 = "From 6am to 9am, higest accidents were on Tuesdays, Wednesdays and Thursdays.\n"
        line3 = "From 10am to 14pm and 19pm to 23pm, accidents were highest on Fridays & Saturdays.\n"
        line4 = "From 15pm to 18pm, accidents were highest on Tuesdays, Wednesdays, Thursdays, and Fridays."
        annotation = line1+line2+line3+line4
        r.text(0, 4500, annotation, color='black', weight='bold', size=13,
               bbox={'facecolor': 'none', 'edgecolor':'blue'})
        r.set_ylim(top=5000)
        sns.move_legend(r, loc=[0.83, 0.55])
        
        fig_filename = f'hourly_trend_{fname_suffix.lower()}.png'
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
                    annotate=True, paletter={'Fatal': 'red', 'Serious': 'orange', 'Slight': 'gray'}, axis=ax, 
                    plot_title='Relationship Between Accident and Casualty Severity')


        line1 = 'Majority of accidents were minor.\nMost casualties sustained minor injuries.'  
        line2 = '\nMost (not all) casualties in FATAL accidents incurred FATAL injuries.'
        line3 = '\nMost casualties in SERIOUS accidents incurred SERIOUS injuries'
        line4 = '\nNo casualty in minor accidents incurred fatal injury'
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

        line1 = "For MINOR/SLIGHT accidents, all casualties sustained only minor injuries.\n"
        line2 = "For SERIOUS accidents, casualties sustained both serious and minor injuries\n"
        line3 = "For FATAL accidents, casualties sustained some slight, serious, and fatal injuries."
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
        cls.plot_column(freq[x], freq['%total'], freq[y], annotate=True, 
                         axis=ax, plot_title=f'Relationship Between {x.capitalize()} and {y.capitalize()}',
                        paletter={'Urban': 'green', 'Rural': 'gray'})

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
        cls.plot_column(freq[x], freq['%total'], freq[y], annotate=True,
                    axis=ax, plot_title=f'Relationship Between {x.capitalize()} and {y.capitalize()}',
                        paletter={'Urban area': 'green', 'Small town': 'yellow', 'Rural': 'gray'})

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
        cls.plot_column(freq[x], freq['%total'], freq[y], annotate=True, 
                         axis=ax, plot_title=f'Relationship Between {x.capitalize()} and {y.capitalize()}',)
        #                 paletter={'Urban': 'green', 'Rural': 'gray'})


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
        cls.plot_column(freq[x], freq['%total'], freq[y], annotate=True, 
                         axis=ax, plot_title=f'Relationship Between {x.capitalize()} and {y.capitalize()}',)
        #                 paletter={'Urban': 'green', 'Rural': 'gray'})

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
        fname = 'line_pedloc_pedmovt.png'
        print(cls.fig_writer(fname, fig));
        
        # 7. show relationship between part_of_day and light_conditions
        # plot column
        x, y = 'light_conditions', 'part_of_day_num'
        #display(var_look['Light Conditions'])
        freq = df[[x, y]].groupby([x, y]).size().sort_index().reset_index()
        freq.columns = freq.columns.astype(str).str.replace('0', 'total_count')

        guide = {1:'morning', 2:'afternoon', 3:'evening', 4:'night'}
        freq['part_of_day'] = freq[y].map(guide)
        freq['%total'] = cls.give_percentage(freq['total_count'])
        ll = dict(var_look['Light Conditions'].to_records(index=False))
        freq[x] = freq.apply(lambda row: ll[row[x]],
                  axis=1)
        # print(freq)

        fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=150)
        cls.plot_bar(freq['total_count'], freq[x], condition_on=freq['part_of_day'], #include_perc=include_perc, 
                         annotate=True, paletter={'morning': 'orange', 'afternoon': 'yellow', 'evening':'gray', 'night':'black'}, 
                         plot_title='Relationship Between Light Conditions and Part of Day', axis=ax)

        line1 = "Interesting to see that 2,082 daylight accidents happended in the night (21pm to 4.59am).\n" 
        line2 = 'However, 70% of them occured in the 5th, 22th, 23rd hour of the day.\n'
        line3 = 'Also astonishing to see 762 morning and afternoon accidents occured in darkness with no lights on.'

        annot = line1+line2+line3#+line4
        ax.text(12000, 2.15, annot, color='black', weight='bold', size=7.5,
                 bbox={'edgecolor':'red', 'facecolor':'none'})
        # sns.move_legend(ax, [0.05, 0.35])
        fname = 'hist_ligcond_pod.png'
        print(cls.fig_writer(fname, fig));

        
        
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
                cls.plot_scatter(freq.index, freq, condition_on=condition_on, plot_title=f'Accident Distribution per {col.capitalize()}', 
                                  x_labe=col, y_labe='Count', axis=ax1)
                cls.plot_box(df[col], plot_title=f'{col.capitalize()} Boxplot', axis=ax2)
                plt.show()
            elif 20 < len(freq.index) <= 30:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), dpi=150)
                cls.plot_hist(df[col], condition_on=condition_on, plot_title=f'Accident Distribution per {col.capitalize()}', 
                                  x_labe=col, y_labe='Count', bins=10, axis=ax1)
                cls.plot_box(df[col], plot_title=f'{col.capitalize()} Boxplot', axis=ax2)
                plt.show()
            else:
                fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=150)
                cls.plot_column(freq.index, freq, plot_title=f'Accident Distribution per {col.capitalize()}', 
                               x_labe=col, y_labe='Count', xy_labe_size=14, axis=ax)
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
    def visualize_severity(accidents, pivot_cols=None, use_as_x='month_name', plot_title='Accident Casualty Outcomes per Month', 
                           xlim=None, savefig=False, fig_filename='outcome_per_month.png'):
        
        cls = RoadAccidents()
        constants = ['casualty_severity', 'severity', 'total_count']
        if not pivot_cols:
            pivot_on = ['month', 'month_name'] + constants
        else:
            if not isinstance(pivot_cols, (list, tuple)):
                pivot_on = [pivot_cols] + constants
                print(pivot_on)
            else:
                pivot_on = pivot_cols + constants
        accidents['severity'] = cls.assign_severity(accidents)
        severity_agg = cls.generate_aggregated_lookup(accidents, include_cols=pivot_on[:-1])
        outcome_agg = severity_agg[pivot_on].groupby(pivot_on[:-1]).sum().sort_index().reset_index()
    #     display(outcome_agg)
        cmap = {'fatal':'black', 'serious':'red', 'minor':'gray'}
        cls.plot_bar(y=outcome_agg[use_as_x], x=outcome_agg[constants[-1]], condition_on=outcome_agg[constants[1]],
                     paletter=cmap, plot_title=plot_title,  annotate=True, xlim=xlim,
                      savefig=savefig, fig_filename=fig_filename)
    
    @staticmethod
    def generate_aggregated_lookup(df, include_cols: list=None):
        """generate a lookup table using a subset of variables
        comprising a total accident count per category.
        Return:
        freq_lookup_df"""
        
        using_cols = ['month', 'month_name', 'quarter', 'season', 'week_num', 'day_of_week',
                    'day_name', 'is_dst', 'is_weekend', 'is_sunrise', 'is_sunset', 'part_of_day_num', 
                    'hour', 'minute', 'day_num']
        
        extra_cols = ['part_of_day', 'inactive_hour', 'season_num']
        all_cols = using_cols + extra_cols

        if include_cols:
            if isinstance(include_cols, (tuple, list)):
                cols_not_included = []
                
                for col in include_cols:
                    if str.lower(col) not in all_cols:
                        cols_not_included.append(col)
            else:
                if str.lower(include_cols) not in all_cols:
                    cols_not_included.append(include_cols)
            using_cols = using_cols + cols_not_included
        aggregate_df = df[using_cols].value_counts().sort_index().reset_index()
        aggregate_df.columns = aggregate_df.columns.astype(str).str.replace('0', 'total_count')
         
        day_parts = {1:'morning', 2:'afternoon', 3:'evening', 4:'night'}
        aggregate_df['part_of_day'] = aggregate_df['part_of_day_num'].apply(lambda x: day_parts[x])
        
        night_times = (0, 1, 2, 3, 4, 5, 6, 22, 23)
        aggregate_df['inactive_hour'] = aggregate_df['hour'].apply(lambda x: 1 if x in night_times else 0)
        
        seasons = {'winter':1, 'spring':2, 'summer':3, 'autumn':4}
        aggregate_df['season_num'] = aggregate_df['season'].apply(lambda x: seasons[x])
        
        #aggregate_df = df[using_cols].value_counts().sort_index().reset_index()
        #aggregate_df.columns = aggregate_df.columns.astype(str).str.replace('0', 'total_count')
        
        x = list(aggregate_df.drop('total_count', axis=1).columns)
        final_cols = x + ['total_count']
        print(final_cols)
        return aggregate_df[final_cols]
                             
    @staticmethod
    def assign_part_of_day(df, in_words=True):
        """get part of day (i.e. morning, afternoon, evening, night) 
        from parameters hr and minuter.
        1. morning: 5am - 11.59am
        2. afternoon: 12pm - 16.59pm
        3. evening: 17pm - 20.59pm
        4. night: 21pm - 4.59am
        Return:
        part_of_day: in words if in_words else integer"""
            
        # morning index
        morn_hrs = (5, 6, 7, 8, 9, 10, 11)
        morn_ind = tuple(df.loc[df['hour'].isin(morn_hrs)].index)
        
        # afternoon index
        aft_hrs = (12, 13, 14, 15, 16)
        aft_ind = tuple(df.loc[df['hour'].isin(aft_hrs)].index)
        
        # evening index
        evn_hrs = (17, 18, 19, 20)
        evn_ind = tuple(df.loc[df['hour'].isin(evn_hrs)].index)
        
        # morning index
        ngh_hrs = (21, 22, 23, 0, 1, 2, 3, 4)
        ngh_ind = tuple(df.loc[df['hour'].isin(ngh_hrs)].index)
        
        part_of_day = pd.Series(index=df.index, name='part_of_day')
        
        if in_words:
            part_of_day.loc[morn_ind] = 'morning'
            part_of_day.loc[aft_ind] = 'afternoon'
            part_of_day.loc[evn_ind] = 'evening'
            part_of_day.loc[ngh_ind] = 'night'
        else:
            part_of_day.loc[morn_ind] = 1
            part_of_day.loc[aft_ind] = 2
            part_of_day.loc[evn_ind] = 3
            part_of_day.loc[ngh_ind] = 4
            
        return part_of_day
                             
    @staticmethod
    def get_month(val: str, in_words: str = False):
        """change datatype of month from words to integer (vice versa)
        """
        
        if not isinstance(val, (int, str)):
            raise TypeError('val must be int or str')
        
        if isinstance(val, int):
            if (0 > val) or (val > 12):
                raise ValueError('val must be between 1 and 12')
        
        guide = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                 'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
        
        if isinstance(val, str):
            if val.capitalize() not in guide.keys():
                raise ValueError('val is wrongly spelt')

        def convert_to_words(val):
            return dict([(v, k) for k, v in guide.items()])[val]

        def convert_to_int(val):
            for k, v in guide.items():
                if val.lower() in k.lower():
                    return v
        if in_words:
            return convert_to_words(val)
        return convert_to_int(val)
        
    @staticmethod
    def compute_full_hour(hour, minute):
        frac_min = minute/60
        return np.round(hour+frac_min, 2)
        
    @staticmethod
    def assign_full_hour(accidents):
        cls = RoadAccidents()
        full_hour = accidents.apply(lambda row: cls.compute_full_hour(row['hour'], row['minute']), axis=1)
        full_hour.name = 'full_hour'
        return full_hour
        
    @staticmethod
    def assign_sunrise_sunset(accidents, rise=True, lower_bound=True, as_boundaries=False):
        
        sunrise_sunset_mn = {1:{'rise_start': '7:40', 'rise_end':'8:06', 'set_start': '16:01', 'set_end': '16:47'},
                         2:{'rise_start': '6:48', 'rise_end':'7:49', 'set_start':'16:49', 'set_end':'17:48'},
                         3:{'rise_start': '5:41', 'rise_end':'6:46', 'set_start':'17:40', 'set_end':'18:29'},
                         4:{'rise_start': '5:35', 'rise_end':'6:36', 'set_start':'19:33', 'set_end':'20:21'},
                         5:{'rise_start': '4:50', 'rise_end':'5:33', 'set_start':'20:23', 'set_end':'21:6'},
                         6:{'rise_start': '4:42', 'rise_end':'4:49', 'set_start':'21:07', 'set_end':'21:21'},
                         7:{'rise_start': '4:47', 'rise_end':'5:22', 'set_start':'20:50', 'set_end':'21:20'},
                         8:{'rise_start': '5:23', 'rise_end':'6:10', 'set_start':'19:49', 'set_end':'20:49'},
                         9:{'rise_start': '6:12', 'rise_end':'6:58', 'set_start':'18:41', 'set_end':'19:47'},
                         10:{'rise_start': '6:44', 'rise_end':'7:42', 'set_start':'16:35', 'set_end':'18:39'},
                         11:{'rise_start': '6:53', 'rise_end':'7:41', 'set_start':'15:55', 'set_end':'16:34'},
                         12:{'rise_start': '7:43', 'rise_end':'8:6', 'set_start':'15:51', 'set_end':'16:00'}}

        cls = RoadAccidents()
        def get_rise_start(accidents):
            start_str = accidents['month'].apply(lambda x: sunrise_sunset_mn[x]['rise_start'])
            start_str.name = 'rise_start'
            start_full_hr = start_str.apply(lambda x: 
                                            cls.compute_full_hour(float(x.split(':')[0]), 
                                                                   float(x.split(':')[1])))

            return start_full_hr
        
        def get_rise_end(accidents):
            end = accidents['month'].apply(lambda x: sunrise_sunset_mn[x]['rise_end'])
            end.name = 'rise_end'
            end_full_hr = end.apply(lambda x: 
                                            cls.compute_full_hour(float(x.split(':')[0]), 
                                                                   float(x.split(':')[1])))

            return end_full_hr
        
        def get_set_start(accidents):
            start_str = accidents['month'].apply(lambda x: sunrise_sunset_mn[x]['set_start'])
            start_str.name = 'set_start'
            start_full_hr = start_str.apply(lambda x: 
                                            cls.compute_full_hour(float(x.split(':')[0]), 
                                                                   float(x.split(':')[1])))

            return start_full_hr
        
        def get_set_end(accidents):
            end = accidents['month'].apply(lambda x: sunrise_sunset_mn[x]['set_end'])
            end.name = 'set_end'
            end_full_hr = end.apply(lambda x: 
                                            cls.compute_full_hour(float(x.split(':')[0]), 
                                                                   float(x.split(':')[1])))

            return end_full_hr
        
        def assign_sunrise(lookup_df):
            func = lambda row: 1 if ((row['full_hour'] >=  row['rise_start']) 
                                     
                                     and (row['full_hour'] <=  row['rise_end'])) else 0
            
            return lookup_df.apply(func, axis=1)
        
        def assign_sunset(lookup_df):
            func = lambda row: 1 if ((row['full_hour'] >=  row['set_start']) 
                                     
                                     and (row['full_hour'] <=  row['set_end'])) else 0
            
            return lookup_df.apply(func, axis=1)
        
        full_hours = cls.assign_full_hour(accidents)
        rise_start = get_rise_start(accidents)
        rise_end = get_rise_end(accidents)
        set_start = get_set_start(accidents)
        set_end = get_set_end(accidents)
        
        rise_set_lookup = pd.concat([full_hours, rise_start, rise_end, set_start, set_end], axis=1)
        if as_boundaries:
            return rise_set_lookup
        sunrise = assign_sunrise(rise_set_lookup)       
        sunrise.name = 'is_sunrise'
        sunset = assign_sunset(rise_set_lookup)
        sunset.name = 'is_sunset'
        return pd.concat([sunrise, sunset], axis=1)
        
    @staticmethod
    def engineer_useful_features(acc_df):
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
        df['part_of_day_num'] = cls.assign_part_of_day(acc_df, in_words=False) 
        
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
        
        #8. derive day_of_week from day_name using dictionary guide
        guide = {'Monday':1, 'Tuesday':2, 'Wednesday':3, 'Thursday':4, 'Friday':5, 'Saturday':6, 'Sunday':7}
        df['day_name'] = df['date'].astype(np.datetime64).dt.day_name()
        df['day_of_week'] = df['day_name'].map(guide)
        
        #9. derive is_weekend from day_name
        guide = {'Sunday': 1, 'Monday': 0, 'Tuesday': 0, 'Wednesday': 0, 'Thursday': 0, 'Friday': 0, 'Saturday': 1}
        df['is_weekend'] = df['day_name'].apply(lambda x: guide[x])
        
        #10. derive seasons from month and day
        df['season'] = cls.assign_seasons(df[['month', 'day']])
        
        #11. derive DST indicator variable
        df['is_dst'] = cls.select_dst_period(df[['month_name', 'day', 'hour']]).astype(int)
        
        #12. derive premier league offseason
        df['is_offseason'] = cls.assign_pl_offseason(df)
        
        #13. derive sunrise and sunset accident indicator
        df[['is_sunrise', 'is_sunset']] = cls.assign_sunrise_sunset(df)
        
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
                           xticklabe_size=16, yticklabe_size=16, annot_size=20, title_size=20, 
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
    def compute_balanced_class_weights(y, class_labels=None):
        """compute a balanced weight per class
        use the output dictionary as argument for class_weight 
        parameter for keras models
        Returns:
        class_weight_coeffs: list of balanced class weights coefficient"""
        
        if class_labels is None:
            class_labels = [0, 1]
            
        cls_weights = s_utils.class_weight.compute_class_weight(class_weight='balanced', classes=class_labels, y=y)
        class_w = {l:w for l, w in zip(class_labels, cls_weights)}
        return class_w
        
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
    def test_hypotheses(selected_data, agg_cols=None, focus_col=None, bigger_set_name='X1', smaller_set_name='X2',
                        bigger_set_vals=None, smaller_set_vals=None, second_condition_col=None, second_condition_val=None,
                        third_condition_col=None, third_condition_val=None, balance_unequal=True):
        """run hypothesis test on selected data for focus_col's bigger_set_name > focus_col's smaller_set_name"""
        
        cls = RoadAccidents()
        if not agg_cols:
            agg_cols = ['hour']
        a_cols = [str.lower(c) for c in agg_cols if c != 'total_count']
        agg_df = cls.generate_aggregated_lookup(selected_data, include_cols=a_cols)
        x_cols = a_cols + ['total_count']
        display(x_cols)
        # total accidents
        total_acc = agg_df[x_cols].groupby(a_cols).sum().reset_index()
        display(total_acc)
        bigger_set = cls.get_accidents_when(total_acc, col1=focus_col, col1_is=bigger_set_vals, 
                                             col2=second_condition_col, col2_is=second_condition_val,
                                            col3=third_condition_col, col3_is=third_condition_val)
        print(f"{bigger_set_name}:")

        display(bigger_set)
        smaller_set = cls.get_accidents_when(total_acc, col1=focus_col, col1_is=smaller_set_vals,
                                             col2=second_condition_col, col2_is=second_condition_val,
                                            col3=third_condition_col, col3_is=third_condition_val)
        print("\n>\n")
        print(f"{smaller_set_name}:")
        display(smaller_set)
        
        cls.report_a_significance(bigger_set['total_count'], smaller_set['total_count'],
                                   X1_name=bigger_set_name, X2_name=smaller_set_name,
                                  balance=balance_unequal)
        
    @staticmethod
    def report_a_significance(X1_set, X2_set, n_deg_freedom=1, X1_name='X1', X2_name='X2', seed=None, balance=True):
        """Test for statistical significant difference between X1_set and X2_set
        at 99% and 95% Confidence.
        X1_set: 1D array of observations
        X2_set: 1D array of observations."""
        
        cls = RoadAccidents()
        
        def get_min_denom(n1, n2):
            return min([n1, n2])
        
        def detect_unequal_sizes(n1, n2):
            """check if both lengths are not the same"""
            return n1 != n2
        
        # to ensure reproducibility
        if not seed:
            seed = 1
        np.random.seed(seed)
        
        samp_sizes = {X1_name: pd.Series(X1_set), 
                      X2_name: pd.Series(X2_set)}
        
        print(f'\n\nHYPOTHESIS TEST FOR:\n{X1_name} > {X2_name}\n')
        
        # use to compare single values
        if len(samp_sizes[X1_name]) == 1:
            total_X1, X1_mean, X1_std = samp_sizes[X1_name].iloc[0], samp_sizes[X1_name].iloc[0], samp_sizes[X1_name].iloc[0]**0.5
            total_X2, X2_mean, X2_std = samp_sizes[X2_name].iloc[0], samp_sizes[X2_name].iloc[0], samp_sizes[X2_name].iloc[0]**0.5
        else:
            X1_size, X2_size = len(X1_set), len(X2_set)
            print(f'ORIGINAL SAMPLE SIZE: \n{X1_name}: {X1_size}\n' +
                  f'{X2_name}: {X2_size}\n\n')
            
            # check if sample sizes are unequal
            if detect_unequal_sizes(X1_size, X2_size):
                print("Unequal Sample Sizes Detected!!\n")
                if balance:
                    print("\n....DOWNSAMPLING RANDOMLY....\n")
                    min_size = get_min_denom(X1_size, X2_size)
                    max_samp_name = [name for name, val in samp_sizes.items() if len(val) != min_size][0]
                    # downsampling: 
                    # randomly generate min_size indexes for max_samp_name
                    rand_indexes = cls.index_generator(len(samp_sizes[max_samp_name]), min_size, random_state=seed)
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
        stdv = np.std(arr).round(4)
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
    def index_generator(sample_size: 'array', n_index=1, random_state=1):
        """Randomly generate n indexes.
        :Return: random_indexes"""

        import random

        def select_from_array(sample_array, n_select=1, random_state=1):
            np.random.seed(random_state)
            return random.choices(population=sample_array, k=n_select)
            
        indexes = range(0, sample_size, 1)

        return select_from_array(indexes, n_index, random_state)
    
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
    def plot_scatter(x, y, condition_on=None, plot_title='Scatter Plot', title_size=14, marker=None, color=None, 
                         paletter='viridis', x_labe=None, y_labe=None, xy_labe_size=8, axis=None, figsize=(8, 4), dpi=200,
                         rotate_xticklabe=False, alpha=None, savefig=False, fig_filename='scatterplot.png'):
            """plot scatter graph on an axis.
            Return: axis """
            
            cls = RoadAccidents()
            
            if not axis:
                fig = plt.figure(figsize=figsize, dpi=dpi)
                axis = sns.scatterplot(x=x, y=y, hue=condition_on, marker=marker, palette=paletter, color=color)
            else:
                sns.scatterplot(x=x, y=y, hue=condition_on, marker=marker,
                                alpha=alpha, ax=axis, palette=paletter, color=color)
            axis.set_title(plot_title, weight='bold', size=title_size)
            if x_labe:
                axis.set_xlabel(str.capitalize(x_labe), weight='bold', size=xy_labe_size)
            if y_labe:
                axis.set_ylabel(str.capitalize(y_labe), weight='bold', size=xy_labe_size)
                
            if savefig:
                print(cls.fig_writer(fig_filename, fig))
            return axis
        
    @staticmethod
    def plot_line(x, y, condition_on=None, plot_title='Line Plot', line_size=None, legend_labe=None, show_legend=False,
                marker=None, color=None,  x_labe=None, y_labe=None, axis=None, figsize=(8, 4), dpi=200,
                savefig=False, fig_filename='lineplot.png'):
        """plot line graph on an axis.
        Return: axis """
        
        cls = RoadAccidents()
        if not axis:
            fig = plt.figure(figsize=figsize, dpi=dpi)
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
        
        if savefig:
            print(cls.fig_writer(fig_filename, fig))
        return axis
                          
    @staticmethod
    def plot_column(x, y, condition_on=None, plot_title='A Column Chart', x_labe=None, y_labe=None, annotate=True,
                color=None, paletter='viridis', conf_intvl=None, include_perc=False, xy_labe_size=8,
                annot_size=6, top_labe_gap=None, h_labe_shift=0.1, top_labe_color='black', bot_labe_color='blue', 
                index_order: bool=True, rotate_xticklabe=False, title_size=15, xlim: tuple=None, ylim: tuple=None, axis=None, 
                figsize=(8, 4), dpi=200, savefig=False, fig_filename='columnplot.png'):
        """plot bar graph on an axis.
        If include_perc is True, then perc_freq must be provided.
        :Return: axis """
        
        cls = RoadAccidents()
        freq_col = pd.Series(y)
        total = freq_col.sum()
        
        if color:
            paletter = None
        if not axis:
            fig = plt.figure(figsize=figsize, dpi=dpi)
            axis = sns.barplot(x=x, y=y, hue=condition_on, 
                              palette=paletter, color=color, ci=conf_intvl)
        else:
            sns.barplot(x=x, y=y, hue=condition_on, ci=conf_intvl, 
                        palette=paletter, color=color, ax=axis)
        
        axis.set_title(plot_title, weight='bold', size=title_size)
        
        if rotate_xticklabe:
                axis.set_xticklabels(axis.get_xticklabels(), rotation=90)
        if x_labe:
            axis.set_xlabel(str.capitalize(x_labe), weight='bold', size=xy_labe_size)
        if y_labe:
            axis.set_ylabel(str.capitalize(y_labe), weight='bold', size=xy_labe_size)
        
        y_range = y.max() - y.min()
        if not top_labe_gap:
            top_labe_gap=y_range/1000
        
        if annotate: 
            cont = axis.containers
            for i in range(len(cont)):
#                 print(len(cont))
                axis.bar_label(axis.containers[i], color=bot_labe_color, size=annot_size,
                                  weight='bold')
                    
                if include_perc:
                    for n, p in enumerate(axis.patches):
                        x, y = p.get_xy()
                        i = p.get_height()
                        perc = round(100 * (i/total), 2)
                        labe = f'{perc}%'
                        top_labe_pos = i+top_labe_gap
                        axis.text(x-h_labe_shift, top_labe_pos, labe, color=top_labe_color, 
                                  size=annot_size, weight='bold')
        
        if xlim:
            axis.set_xlim(left=xlim[0], right=xlim[1])
        
        if ylim:
            axis.set_ylim(bottom=ylim[0], top=ylim[1])
            
        if savefig:
            print(cls.fig_writer(fig_filename, fig))
        return axis
        
    @staticmethod
    def plot_bar(x, y, condition_on=None, plot_title='A Bar Chart', title_size=15, x_labe=None, y_labe=None,
             color=None, paletter='viridis', conf_intvl=None, include_perc=False, perc_freq=None, annot_size=6,
             bot_labe_gap=10, top_labe_gap=10, v_labe_shift=0.4, top_labe_color='black', xy_labe_size=8,
             bot_labe_color='blue', index_order: bool=True, rotate_yticklabe=False, annotate=False, axis=None,
             xlim=None, figsize=(8, 4), dpi=150, savefig=False, fig_filename='barplot.png'):
        """plot bar graph on an axis.
        If include_perc is True, then perc_freq must be provided.
        :Return: axis """
        
        cls = RoadAccidents()
        freq_col = x

        if color:
            paletter = None

        if not axis:
            fig = plt.figure(figsize=figsize, dpi=dpi)
            axis = sns.barplot(x=x, y=y, hue=condition_on, orient='h',
                              palette=paletter, color=color, ci=conf_intvl)
        else:
            sns.barplot(x=x, y=y, hue=condition_on, ci=conf_intvl, 
                        palette=paletter, color=color, orient='h', ax=axis)

        axis.set_title(plot_title, weight='bold', size=title_size,)

        if rotate_yticklabe:
            axis.set_yticklabels(axis.get_yticklabels(), rotation=90)
        if x_labe:
            axis.set_xlabel(str.capitalize(x_labe), weight='bold', size=xy_labe_size)
        if y_labe:
            axis.set_ylabel(str.capitalize(y_labe), weight='bold', size=xy_labe_size)
        if xlim:
            axis.set_xlim(left=xlim[0], right=xlim[1])
        if annotate: 
            cont = axis.containers
#                 print(len(cont))
            for i in range(len(cont)):
                axis.bar_label(axis.containers[i], color=bot_labe_color, size=annot_size,
                weight='bold')
                if include_perc and perc_freq is not None:
                    labe = f'({perc_freq.iloc[i]}%)'
                    axis.text(freq_col.iloc[i]+top_labe_gap, i-v_labe_shift, labe,
                              color=top_labe_color, size=annot_size, weight='bold')
                              
        if savefig:
            print(cls.fig_writer(fig_filename, fig))
        return axis
    
    @staticmethod
    def plot_box(x=None, y=None, condition_on=None, plot_title="A Boxplot", orientation='horizontal', 
                 x_labe=None, y_labe=None, axis=None, figsize=(8, 4), dpi=150, savefig=False, fig_filename='boxplot.png'):
        """A box distribution plot on an axis.
        Return: axis"""
        
        cls = RoadAccidents()
        
        if (not isinstance(x, (pd.Series, np.ndarray)) and x is not None):
            raise TypeError("x must be a pandas series or numpy array")
        elif (not isinstance(y, (pd.Series, np.ndarray)) and y is not None):
            raise TypeError("y must be a pandas series or numpy array")
            
        if not axis:
            fig = plt.figure(figsize=figsize, dpi=dpi)
            axis = sns.boxplot(x=x, y=y, hue=condition_on, orient=orientation)
        else:
            sns.boxplot(x=x, y=y, hue=condition_on, orient=orientation, ax=axis)
        axis.set_title(plot_title, weight='bold')
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
        
        cls = RoadAccidents()
        
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
            fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
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
    def plot_diff(left_side, right_side, left_legend='Left', right_legend='right', xlim=None, ylim=None,
              plot_title='Comparison Plot', left_legend_color='white', right_legend_color='white', 
              left_labe_shift=0, left_vlabe_shift=0, right_labe_shift=0, right_vlabe_shift=0, l_bbox_color='orange', 
              r_bbox_color='blue', fig_w=6, fig_h=8, dpi=200, savefig=False, fig_filename='comparison_plot.png'):
        """Comparison view of left values vs right values."""
            
        cls = RoadAccidents()
        diff = right_side - left_side
        color_mapping = {i: r_bbox_color if v >= 0 else l_bbox_color for i, v in zip(diff.index, diff)}
        #         print(color_mapping)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

        cls.plot_bar(x=diff, y=diff.index, y_labe='Hour', plot_title=plot_title, x_labe='Total Accidents',
                      bot_labe_color='black', annotate=True, v_labe_shift=0.05, paletter=color_mapping, 
                      axis=ax)

        left_pos = diff.min() + diff.min()/2
        right_pos = diff.max() + diff.max()/2
        
        if not xlim:
            xlim = (left_pos, right_pos)
        ax.set_xlim(xlim[0], xlim[1])

        labe = left_legend
        min_ind = left_side.loc[left_side == left_side.min()].index[0]
        x_pos = diff.min()/2
        ax.text(x_pos-left_labe_shift, min_ind-left_vlabe_shift, labe, color=left_legend_color, 
                weight='bold', bbox={'facecolor':l_bbox_color})

        labe = right_legend
        min_ind = right_side.loc[right_side == right_side.min()].index[0]
        x_pos = diff.max()/2
        ax.text(x_pos+right_labe_shift, min_ind+right_vlabe_shift, labe, color=right_legend_color, 
                weight='bold', bbox={'facecolor':r_bbox_color})
        plt.show()
        # save figure to filesystem as png
        if savefig:
            fname = fig_filename
            print(cls.fig_writer(fname, fig))
            
    @staticmethod
    def plot_pyramid(left_side, right_side, left_legend='Left', right_legend='right', xlim=None, ylim=None,
                     plot_title='Pyramid Plot', left_legend_color='white', right_legend_color='white', 
                     left_labe_shift=0, right_labe_shift=0, l_bbox_color='orange', r_bbox_color='blue', fig_w=6,
                     fig_h=8, savefig=False, fig_filename='pyramid_plot.png', dpi=200):
        """Pyramid view of negative values vs positive values."""
        
        cls = RoadAccidents()
        negative_side = -left_side
        positive_side = right_side
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

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
        
        if not xlim:
            xlim = (left_pos, right_pos)
        ax.set_xlim(xlim[0], xlim[1])

        labe = left_legend
        min_ind = negative_side.loc[negative_side == negative_side.max()].index[0]
        x_pos = (abs(negative_side.min()) - abs(negative_side.max()))/2
        ax.text(-x_pos-left_labe_shift, min_ind, labe, color=left_legend_color, 
                weight='bold', bbox={'facecolor':l_bbox_color})

        labe = right_legend
        max_ind = positive_side.loc[positive_side == positive_side.min()].index[0]
        x_pos = (positive_side.max() - positive_side.min())/2
        ax.text(x_pos+right_labe_shift, max_ind, labe, color=right_legend_color, 
                weight='bold', bbox={'facecolor':r_bbox_color})
        plt.show()
        # save figure to filesystem as png
        if savefig:
            fname = fig_filename
            print(cls.fig_writer(fname, fig))
        
    @staticmethod
    def visualize_trends(database: pd.DataFrame, fig_filename_suffix='fig'):
        """plots showing accident counts per month, week, day, day_name, hour, and minute;
        comparison plots"""
        
        cls = RoadAccidents()
        def plot_trend_line(x, y, plot_title='Line Plot', title_size=10, x_labe='x axis', y_labe='y axis',
                            text_size=6, text_color='white', bbox_color='black', fig_filename='line_plot.png', 
                             xlim=(0, 10), fig_w=6, fig_h=8):
        
            # set background style
            sns.set_style('darkgrid')
            # compute average 
            avg = y.mean().round(2)
            # boundaries for average line marker
            avg_line_pos=(xlim[0]-2, xlim[1]+2)

            # figure dimensions
            fig, ax1 = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
            # average line
            cls.plot_line([avg_line_pos[0], avg_line_pos[1]], [avg, avg], 
                          color='black', line_size=1, axis=ax1)
            # trend line
            cls.plot_line(x, y, marker='o', color='red', 
                          y_labe=y_labe, x_labe=x_labe, axis=ax1)
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
            fname = fig_filename
            print(cls.fig_writer(fname, fig))
            
        def plot_distribution(x, y, plot_title='Line Plot', x_labe='x axis', y_labe='y axis', include_perc=False,
                              text_size=6, text_color='white', bbox_color='black', fig_filename='distribution_plot.png', 
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
            fig, ax1 = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
            # average line
            ax1 = cls.plot_line([avg_line_pos[0], avg_line_pos[1]], [avg, avg], 
                                color='black', line_size=1, axis=ax1)
            # frequency plot
            y_range = y.max() - y.min()
            if include_perc:
                perc_freq = (100*y/sum(y)).round(2)
                cls.plot_column(x, y, plot_title=plot_title, include_perc=include_perc,
                              top_labe_gap=y_range/30,  
                                 paletter=color_mapping,x_labe='Hour', y_labe='Total Count', axis=ax1)
            else:    
                cls.plot_column(x, y, plot_title=plot_title, include_perc=include_perc,
                                 paletter=color_mapping,x_labe='Hour', top_labe_gap=y_range/30, 
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
            fname = fig_filename
            print(cls.fig_writer(fname, fig))
            
        def plot_dow_distribution(left_side, right_side, lplot_title='Line Plot', rplot_title='Line Plot', left_xlabe='x axis', left_ylabe='y axis',
                                  right_xlabe='x axis', right_ylabe='y axis', include_perc=False, text_size=6, 
                                  text_color='white', bbox_color='black', fig_filename='distribution_plot.png', 
                                   xlim=(0, 8), fig_w=6, fig_h=8):
            """plot double axes distribution plot for days of week."""
    
            left_x, left_y = left_side['day_name'], left_side['total_count']
            right_x, right_y = right_side['day_name'], right_side['total_count']

            avg = left_y.mean().round(2)
            # boundaries of average line marker
            left_avgline_pos=(xlim[0]-2, xlim[1]+2)
            # color_mapping: red for above average, gray for below average
            color_mapping = {i: 'red' if v >= avg else 'gray' for i, v in zip(left_x, left_y)}

            fig, (left, right) = plt.subplots(1, 2, figsize=(fig_w, fig_h), dpi=200)

            # average line
            cls.plot_line([left_avgline_pos[0], left_avgline_pos[1]], [avg, avg],
                           color='black', line_size=1, axis=left)
            # frequency plot
            y_range = left_y.max() - left_y.min()
            if include_perc:
                perc_freq = (100*left_y/sum(left_y)).round(2)
                cls.plot_column(left_x, left_y, plot_title=lplot_title, include_perc=include_perc,
                              top_labe_gap=y_range/30, paletter=color_mapping, x_labe=left_xlabe, y_labe=left_ylabe, axis=left, 
                                 rotate_xticklabe=True)
            else:    
                cls.plot_column(left_x, left_y, plot_title=lplot_title, include_perc=include_perc,
                                 paletter=color_mapping, x_labe=left_xlabe, top_labe_gap=y_range/30, 
                                 y_labe=left_ylabe, axis=left, rotate_xticklabe=True)


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
                              top_labe_gap=y_range/30, paletter=color_mapping, x_labe=right_xlabe, 
                             y_labe=right_ylabe, axis=right, rotate_xticklabe=True)
            else:    
                cls.plot_column(right_x, right_y, plot_title=rplot_title, include_perc=include_perc,
                                  paletter=color_mapping, x_labe=right_xlabe, y_labe=right_ylabe, axis=right,
                                 top_labe_gap=y_range/30, rotate_xticklabe=True)

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
            fname = fig_filename
            print(cls.fig_writer(fname, fig))
            
            
        
        # 1. create frequency table for easier selection
        general_agg = cls.generate_aggregated_lookup(database)
    
        # 1a. Quarterly column
        cols = ['quarter', 'total_count']
        total_acc_q = general_agg[cols].groupby(cols[0]).sum().round(2).reset_index()
        tot = total_acc_q['total_count'].sum()
        perc = total_acc_q['total_count'].apply(lambda x: 100*x/tot).round(2)
    #     print(total_acc_mn)
        cls.plot_column(total_acc_q['quarter'], y=total_acc_q['total_count'], plot_title=f'{str.capitalize(fig_filename_suffix)} Accidents Trend per Quarter',
                        x_labe='quarter', y_labe='total accident', annot_size=6, savefig=True, include_perc=False,  top_labe_gap=3000, 
                        h_labe_shift=-0.32, ylim=(0, 80000), fig_filename=f'hist_quarterly{fig_filename_suffix.lower()}.png')
                        
        # 1b. Quarterly line
        plot_trend_line(total_acc_q['quarter'], y=total_acc_q['total_count'], plot_title=f'{str.capitalize(fig_filename_suffix)} Accidents Trend per Quarter',
                        x_labe='quarter', y_labe='total accident', xlim=(0, 5), fig_filename=f'line_quarterly_{fig_filename_suffix.lower()}.png')
        
        # 1c. Seasonal
        cols = ['season_num', 'season', 'total_count']
        total_acc_ssn = general_agg[cols].groupby(cols[:-1]).sum().round(2).sort_index().reset_index()
        tot = total_acc_ssn['total_count'].sum()
        perc = total_acc_ssn['total_count'].apply(lambda x: 100*x/tot).round(2)

        cls.plot_column(total_acc_ssn['season'], y=total_acc_ssn['total_count'], plot_title=f'{str.capitalize(fig_filename_suffix)} Accidents Trend per Season',
                        x_labe='season', y_labe='total accident', annot_size=7, savefig=True, include_perc=False,  top_labe_gap=3000, 
                        h_labe_shift=-0.32, ylim=(0, 80000), fig_filename=f'seasonal_accidents_{fig_filename_suffix.lower()}.png')
        
        # 2. Monthly Accidents
        cols = ['month', 'total_count']
        total_acc_mn = general_agg[cols].groupby(cols[0]).sum().round(2)
    #     print(total_acc_mn)
        
        # 2b. visualize 
        plot_trend_line(total_acc_mn.index, y=total_acc_mn['total_count'], plot_title=f'{str.capitalize(fig_filename_suffix)} Accidents Trend per Month',
                        x_labe='month', y_labe='total accident', text_size=6, text_color='white',
                       bbox_color='black', fig_filename=f'line_monthly_{fig_filename_suffix.lower()}.png', 
                        xlim=(0, 13))
        
        # 3. Weekly Accidents
        cols = ['week_num', 'total_count']
        total_acc_wk = general_agg[cols].groupby(cols[0]).sum()
    #     print(total_acc_wk)
        
        # 3b. visualize 
        plot_trend_line(total_acc_wk.index, y=total_acc_wk['total_count'], title_size=18,
                        plot_title=f'{str.capitalize(fig_filename_suffix)} Accidents Trend per Week of Year',
                        x_labe='week_num', y_labe='total accident', text_size=6, text_color='white', bbox_color='black',
                        fig_filename=f'line_weekly_{fig_filename_suffix.lower()}.png',
                        xlim=(0, 53), fig_h=6, fig_w=15)
        
        # 4. Daily Accidents
        cols = ['day_num', 'total_count']
        total_acc_dy = general_agg[cols].groupby(cols[0]).sum()
    #     print(total_acc_wk)
        
        # 4b. visualize 
        plot_trend_line(total_acc_dy.index, y=total_acc_dy['total_count'], plot_title=f'{str.capitalize(fig_filename_suffix)} Accidents Trend per Day in Year',
                        title_size=25, x_labe='day_num', y_labe='total accident', text_size=8, text_color='white',
                       bbox_color='black', fig_filename=f'line_daily_{fig_filename_suffix.lower()}.png', 
                         xlim=(0, 370), fig_h=8, fig_w=20)
        
        # 5. Hourly Accidents
        cols = ['hour', 'total_count']
        total_acc_hr = general_agg[cols].groupby(cols[0]).sum()
    #     print(total_acc_wk)
        
        # 5b. visualize 
        plot_trend_line(total_acc_hr.index, y=total_acc_hr['total_count'], 
                        plot_title=f'{str.capitalize(fig_filename_suffix)} Accidents Trend per Hour',
                        x_labe='hour', y_labe='total accident', text_size=6, text_color='white',
                       bbox_color='black', fig_filename=f'line_hourly_{fig_filename_suffix.lower()}.png', 
                         xlim=(-0.5, 25))
        
        # 6. Minutely Accidents
        cols = ['minute', 'total_count']
        total_acc_min = general_agg[cols].groupby(cols[0]).sum()
    #     print(total_acc_wk)
        
        # 6b. visualize 
        plot_trend_line(total_acc_min.index, y=total_acc_min['total_count'], plot_title=f'{str.capitalize(fig_filename_suffix)} Accidents Trend per Minute of Hour',
                        x_labe='minute', y_labe='total accident', text_size=6, text_color='white', title_size=25,
                       bbox_color='black', fig_filename=f'line_minutely_{fig_filename_suffix.lower()}.png', 
                         xlim=(0, 60), fig_h=8, fig_w=12)
        
        # 7. visualize frequency of accidents per hour
        plot_distribution(total_acc_hr.index, y=total_acc_hr['total_count'], plot_title=f'{str.capitalize(fig_filename_suffix)} Accidents Trend per Hour',
                        x_labe='hour', y_labe='total accident', text_size=6, text_color='white',
                       bbox_color='black', fig_filename=f'hist_hourly_{fig_filename_suffix.lower()}.png',
                           xlim=(-1, 24), include_perc=False, fig_w=10, fig_h=6)
        
        # 8. Difference between am vs pm accident count
        am_acc = total_acc_hr.loc[total_acc_hr.reset_index().apply(lambda row: row['hour'] in range(12), 
                                                                        axis=1)].reset_index()
        pm_acc = total_acc_hr.loc[total_acc_hr.reset_index().apply(lambda row: row['hour'] in range(12, 24), 
                                                                   axis=1)].reset_index()
        # 8b. pyramid view
        cls.plot_pyramid(am_acc['total_count'], pm_acc['total_count'], plot_title=f'{str.capitalize(fig_filename_suffix)} AM VS PM Accidents', left_legend='AM', 
                     right_legend='PM', fig_w=12, fig_filename=f'am_pm_pyramid_{fig_filename_suffix.lower()}.png')                                                     
        
        # 8c. comparison view
        cls.plot_diff(am_acc['total_count'], pm_acc['total_count'], plot_title=f'{str.capitalize(fig_filename_suffix)} Differences Between AM and PM Accidents', 
                   left_legend='AM', right_legend='PM', fig_filename=f'am_pm_comparison_{fig_filename_suffix.lower()}.png', fig_w=12)
        
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
                              fig_filename=f'dow_distribution_{fig_filename_suffix.lower()}.png')
        
        
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
                                    annotate=annotate, annot_size=annot_size, top_labe_gap=y_range/30, 
                                    color='brown', figsize=fig_size, dpi=dpi, axis=ax)
                else:
                    cls.plot_column(x=null_cols.index, y=null_cols, plot_title=plot_title, rotate_xticklabe=True, 
                                    x_labe='Column Names', y_labe='Number of Missing Values', annotate=annotate, annot_size=annot_size, 
                                    color='brown', figsize=fig_size, dpi=dpi, axis=ax)
            
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