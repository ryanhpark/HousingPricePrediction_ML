from sklearn.preprocessing import OrdinalEncoder
import category_encoders as ce
import pandas as pd

def encode(train, test):
    '''
    This function has 2 parts: encoding ordinal and nominal categories.
    '''
    
    # Joining both data frames. Have to remove SalePrice from the train df because it is not in the test df
    frames = [train, test]
    new_df = pd.concat(frames, sort=False)
    
    # Save how many rows train/test have
    train_len = len(train.index)
    test_len = len(test.index)
    
    # The following are ordinal categorial features
    ordinalEnc = ["OverallQual","OverallCond","ExterQual","ExterCond","BsmtQual","BsmtCond","BsmtExposure",\
             "BsmtFinType1","HeatingQC","KitchenQual","GarageQual","GarageCond"]
    
    # The following are nominal categorial features
    nominalEnc = ["MSZoning","LotShape","LandContour","LotConfig","Neighborhood","Condition1","BldgType",\
              "HouseStyle","RoofStyle","Exterior1st","Exterior2nd","MasVnrType","Foundation",\
             "CentralAir","Electrical","Functional","GarageType","GarageFinish","PavedDrive","MoSold",\
             "SaleType","SaleCondition"]
    
    # The following function maps each category into a number category. If none is found from the list, 0 is assigned
    def custom_ordinal_encode(ordEncod):
        encoding = {'EX': 5, 'GO': 4, 'AA': 3, 'Avg':2, 'BA':1, \
               'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa':2, 'Po':1, \
               'Av': 3, 'Mn': 2, 'No': 1, \
               'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec':3, 'LwQ':2, 'Unf':1}
        return encoding.get(ordEncod, 0)
    
    # Encode ordinal categorical features
    for col in ordinalEnc:
        encoder_grade = ce.OrdinalEncoder(mapping=[{'col': col, 'mapping': custom_ordinal_encode}], return_df=True)
        new_df = encoder_grade.fit_transform(new_df)

    # Encode nominal categorical features
    enc = OrdinalEncoder()
    new_df[nominalEnc] = enc.fit_transform(new_df[nominalEnc])
    
    # Separate train and test df again with the new feature values. Also, drop "SalePrice" from test
    train_encoded = new_df.head(train_len)
    test_encoded = new_df.tail(test_len)
    test_encoded = test_encoded.drop(columns=['SalePrice'])
    
    
    return train_encoded, test_encoded