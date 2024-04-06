# Import the datasets as pandas dataframes
df_FD001 = pd.read_csv('data/train_FD001.txt', sep='\s+', header=None)
df_FD003 = pd.read_csv('data/train_FD003.txt', sep='\s+', header=None)

df_FD001=df_FD001.rename(columns={0:'unit_nr',1:'time_cycles',2:'setting_1',3:'setting_2',
                                  4:'setting_3',5:'sensor_1',6:'sensor_2',7:'sensor_3',8:'sensor_4',
                                  9:'sensor_5',10:'sensor_6',11:'sensor_7',12:'sensor_8',
                                  13:'sensor_9',14:'sensor_10',15:'sensor_11',16:'sensor_12',
                                  17:'sensor_13',18:'sensor_14',19:'sensor_15',20:'sensor_16',
                                  21:'sensor_17',22:'sensor_18',23:'sensor_19',24:'sensor_20',
                                  25:'sensor_21'})

df_FD003=df_FD003.rename(columns={0:'unit_nr',1:'time_cycles',2:'setting_1',3:'setting_2',
                                  4:'setting_3',5:'sensor_1',6:'sensor_2',7:'sensor_3',8:'sensor_4',
                                  9:'sensor_5',10:'sensor_6',11:'sensor_7',12:'sensor_8',
                                  13:'sensor_9',14:'sensor_10',15:'sensor_11',16:'sensor_12',
                                  17:'sensor_13',18:'sensor_14',19:'sensor_15',20:'sensor_16',
                                  21:'sensor_17',22:'sensor_18',23:'sensor_19',24:'sensor_20',
                                  25:'sensor_21'})


df_FD001['dataset'] = 1
df_FD003['dataset'] = 3

df_FD001 = add_remaining_useful_life(df_FD001)
df_FD003 = add_remaining_useful_life(df_FD003)


# Drop the columns that are not useful for clustering
df_FD001 = StandardScaler().fit_transform(df_FD001.groupby('unit_nr').last().reset_index().drop(columns=['unit_nr', 'time_cycles', 'setting_1','setting_2','setting_3','dataset', 'RUL']))
df_FD003 = StandardScaler().fit_transform(df_FD003.groupby('unit_nr').last().reset_index().drop(columns=['unit_nr', 'time_cycles', 'setting_1','setting_2','setting_3','dataset', 'RUL']))




st = np.concatenate([df_FD001, df_FD001])