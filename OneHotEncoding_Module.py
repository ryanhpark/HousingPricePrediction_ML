from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.pipeline import Pipeline

class Dummify():
    '''
    One-Hot encoder dummifies the train and test sets together to resolve the mismatch between them
    Arguments:
        df1 : train set after feature engineering
        df2 : test set after feature engineering
    Returns: Two dummified dataframes, train and test sets
    '''
    def __init__(self, columns=None):
        self.columns = columns

    def transform(df1, df2):
        # Group the features by dtypes
        train_obj = df1.select_dtypes("object")
        train_num = df1.select_dtypes(["float64","int64"])
        test_obj = df2.select_dtypes("object")
        test_num = df2.select_dtypes(["float64","int64"])

        # Apply OneHotEncoder
        encoder = OneHotEncoder(categories = "auto",drop = 'first',sparse = False)
        train_obj_enc = encoder.fit_transform(train_obj)
        test_obj_enc = encoder.transform(test_obj)
        column_name = encoder.get_feature_names(train_obj.columns.tolist())
        
        # Combine the object and numeric features for train set
        train_df =  pd.DataFrame(train_obj_enc, columns= column_name)
        train_df.set_index(train_num.index, inplace = True)
        train_final = pd.concat([train_df, train_num], axis = 1)
        
        # Combine the object and numeric features for test set
        test_df =  pd.DataFrame(test_obj_enc, columns= column_name)
        test_df.set_index(test_num.index, inplace = True)
        test_final = pd.concat([test_df, test_num], axis = 1)

        return train_final, test_final
    
    def fit(self, X, y=None, **fit_params):
        return self