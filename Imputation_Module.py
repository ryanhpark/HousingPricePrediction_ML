import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer



train_raw = pd.read_csv('../train.csv')
test_raw = pd.read_csv('../test.csv')

train = train_raw.copy()
test = test_raw.copy()



def impute_data():
    '''
    Runs multiple functions to impute training and test datasets.
    
    Accepts:
        none.
        You must have 'train' and 'test' variables containing dataframes in your global variables.
    
    Returns:
        train and test dfs altered in place.
    '''

    def set_df_index(*dfs):
        '''
        Sets the index to the Id column.

        Arguments:
            dfs: One or more dataframes.

        Returns:
            df(s) altered in place.
        '''    

        for df in dfs:
            df.set_index('Id', inplace=True)

    set_df_index(train, test)        

    def num_to_cat_variable(*dfs):
        '''
        Converts columns containing date keywords into dtype 'category'.

        Arguments:
            dfs: One or more dataframes.

        Returns:
            df(s) altered in place.
        '''
        for df in dfs:
            date_cols = list(df.columns[df.columns.str.contains('Year')
                                       | df.columns.str.contains('Yr')
                                       | df.columns.str.contains('Month')                                
                                       | df.columns.str.contains('Mo')])

            df[date_cols] = df[date_cols].astype('str')
            df[date_cols] = df[date_cols].astype('category')
            df[date_cols] = df[date_cols].apply(lambda x: x.cat.add_categories('N/A'))

    num_to_cat_variable(train, test)        

    def impute_lot_frontage(*dfs):
        '''
        Fills the missing values of the LotFrontage column to the means of
        LotFrontage grouped by Neighborhood from the training dataset.

        Arguments:
            dfs: One or more dataframes.

        Returns:
            df(s) altered in place.
        '''

        for df in dfs:
            train_hood_means = dict(train_raw.groupby('Neighborhood').LotFrontage.mean())
            df.LotFrontage = df.LotFrontage.fillna(df.Neighborhood.map(train_hood_means))

    impute_lot_frontage(train, test)        

    def missing_val_info(*dfs):
        '''
        Prints the sum of rows with missing values and the names of columns
        containing NaNs with the sum of their NaN values.

        Arguments:
            dfs: One or more dataframes.

        Returns:
            Printed output.
        '''

        for df in dfs:   
            print('Number of rows with NaN:', len(df[df.isna().any(axis=1)]), '\n')
            cols_na = df.loc[:, df.isna().any()] # df with only columns that have missing values

            if (len(df[df.isna().any(axis=1)]) > 0):
                print('Columns with NaN:\n', cols_na.isna().sum())

            print('-'*30)

#     missing_val_info(train, test)        

    def drop_cols_majority_nan(*dfs):
        '''
        Removes columns that have more than 90% missing values.

        Arguments:
            dfs: One or more dataframes.

        Returns:
            df(s) altered in place.
        '''

        for df in dfs:
            drop_thresh = df.shape[0] * 0.9
            df = df.dropna(axis=1, how='all', thresh=drop_thresh, inplace=True)

    drop_cols_majority_nan(train, test)   

    def impute_not_missing(df, cols):
        '''
        Fills NaN values with the string 'N/A' or the integer 0.

        Arguments:
            df: A dataframe.
            cols: A list of column names as strings.

        Returns:
            df altered in place.
        '''

        cat_dtypes = ['object', 'category']
        num_dtypes = ['int', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16',
                      'uint32', 'uint64', 'float', 'float16', 'float32', 'float64']

        cat_cols = df[cols].select_dtypes(cat_dtypes).columns.tolist()
        num_cols = df[cols].select_dtypes(num_dtypes).columns.tolist()

        df.fillna({x:'N/A' for x in cat_cols}, inplace=True)
        df.fillna({x:0 for x in num_cols}, inplace=True)

    gar_cols = list(train.columns[train.columns.str.contains('Garage')])
    impute_not_missing(train, gar_cols)
    impute_not_missing(test, gar_cols)    

    # If observation has TotalBsmtSF > 0, NaN in any Bsmt column != No Basement
    def impute_bsmt(df, cols):
        '''
        Fills NaN values with the string 'N/A' or the integer 0, excluding cols
        that have TotalBsmtSF > 0.

        Arguments:
            df: A dataframe.
            cols: A list of column names as strings.

        Returns:
            df altered in place.
        '''

        cat_dtypes = ['object', 'category']
        num_dtypes = ['int', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16',
                      'uint32', 'uint64', 'float', 'float16', 'float32', 'float64']

        cat_cols = df[cols].select_dtypes(cat_dtypes).columns.tolist()
        num_cols = df[cols].select_dtypes(num_dtypes).columns.tolist()

        # Store rows where NaN != No Basement
        to_drop = df[cols][(df[cols].isna().any(axis=1)) & (df[cols].TotalBsmtSF > 0)]
        dropped = df[cols].drop(to_drop.index)

        # After dropping rows, fill NaNs 
        dropped.fillna({x:'N/A' for x in cat_cols}, inplace=True)
        dropped.fillna({x:0 for x in num_cols}, inplace=True)

        df[cols] = pd.concat([dropped, to_drop])

    bsmt = list(train.columns[train.columns.str.contains('Bsmt')])
    impute_bsmt(train, bsmt)
    impute_bsmt(test, bsmt)    

    def impute_mean_mode(*dfs):
        '''
        Fills NaN values with the mean or the mode of each column.

        Arguments:
            dfs: One or more dataframes.

        Returns:
            df(s) altered in place.
        '''

        for df in dfs:
            cat_dtypes = ['object', 'category']
            num_dtypes = ['int', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16',
                          'uint32', 'uint64', 'float', 'float16', 'float32', 'float64']

            cat_cols = df.select_dtypes(cat_dtypes).columns.tolist()
            num_cols = df.select_dtypes(num_dtypes).columns.tolist()

            imp_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
            imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')

            df[cat_cols] = imp_mode.fit_transform(df[cat_cols])
            df[num_cols] = imp_mean.fit_transform(df[num_cols])

    impute_mean_mode(train, test)        

#     missing_val_info(train, test)
    
    return train, test