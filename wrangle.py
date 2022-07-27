import pandas as pd
import numpy as np
import os
from env import get_db_url

from sklearn.model_selection import train_test_split

def zillow_data():
    filename = 'zillow.csv'

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        zillow_df = pd.read_sql('''select bedroomcnt, 
                                        bathroomcnt,
                                         calculatedfinishedsquarefeet,
                                          taxvaluedollarcnt,
                                           yearbuilt,
                                            taxamount,
                                             fips,
                                               lotsizesquarefeet,
                                                 poolcnt,
                                                  regionidcounty,
                                                   garagecarcnt
                                              from properties_2017
                                              join propertylandusetype as plut using (propertylandusetypeid) 
                                              join predictions_2017 using (parcelid)
                                              where plut.propertylandusedesc in ("Single Family Residential" , "Inferred Single Family Residential")
                                              and transactiondate like "2017%%";''', get_db_url('zillow'))
        zillow_df.to_csv(filename)

        return zillow_df


def clean_zillow(df):

    #Making pools boolean
    df['poolcnt'] = np.where((df['poolcnt'] == 1.0) , True , False)
    df['garagecarcnt'] = np.where((df['garagecarcnt'] >= 1.0) , df['garagecarcnt'] , 0)  


    #Drop the columns with null values because they only make up about 0.5% and would have less impact on model than imputing value
    df = df.dropna()
    
    # Rename Columns and assigning data type if needed
    df["fed_code"] = df["fips"].astype(int)
    df["year_built"] = df["yearbuilt"].astype(int)
    df["beds"] = df["bedroomcnt"].astype(int)    
    df["home_value"] = df["taxvaluedollarcnt"].astype(float)
    df["sq_ft"] = df["calculatedfinishedsquarefeet"].astype(float)
    df["baths"] = df["bathroomcnt"]
    df["lot_size"] = df["lotsizesquarefeet"]
    df["pools"] = df["poolcnt"]
    df["garages"] = df["garagecarcnt"]
  

    #Feature engineering new variables to combat multicolinearty and test to see if new features help the model

    df['pool_encoded'] = df.pools.map({True:1, False:0})
    
    df['bed_bath_ratio'] = df['beds'] / df['baths']
    df['overall_size'] = df['sq_ft'] + df['lot_size']
    df['house_age'] = 2017 - df['year_built']
    
    # making dummies and encoded values to help machine learning
    # dummy_df = pd.get_dummies(df[['garages']], dummy_na=False,drop_first=False)

    # df = pd.concat([df, dummy_df], axis=1)


    #Deleting duplicate rows
    df = df.drop(columns=['fips', 'yearbuilt', 'bedroomcnt', 'taxvaluedollarcnt', 'calculatedfinishedsquarefeet', 'bathroomcnt', 'poolcnt', 'lotsizesquarefeet', 'year_built', 'garagecarcnt'])
    
    return df







def handle_outliers(df):
    """Manually handle outliers that do not represent properties likely for 99% of buyers and zillow visitors"""
    df = df[df.baths <= 6]

    df =df[df.baths > 0]
    
    df = df[df.beds <= 6]

    df =df[df.beds > 0]

    df = df[df.sq_ft <= 10_000]

    df = df[df.sq_ft > 700]

    df = df[df.home_value < 2_000_000]

    df = df[df.lot_size <=100_000]

    return df



def wrangle_zillow():

    df = zillow_data()

    df = clean_zillow(df)

    df = handle_outliers(df)

    # df.to_csv("zillow.csv", index=False)

    return df

# split the data before fitting scalers 
def zillow_split():
    
    z_train, z_test = train_test_split(df, train_size=0.8, random_state=123)
    z_train, z_validate = train_test_split(z_train, train_size=0.7, random_state=123)

    return train, validate, test

cols_to_scale = ['year_built', 'beds','sq_ft', 'baths']

def MinMax_scaler(z_train, z_validate, z_test):
 
    scaler = MinMaxScaler().fit(z_train)
    z_train_scaled = pd.DataFrame(scaler.transform(z_train[cols_to_scale]), index=z_train.index)
    z_validate_scaled = pd.DataFrame(scaler.transform(z_validate[cols_to_scale]), index=z_validate.index)
    z_test_scaled = pd.DataFrame(scaler.transform(z_test[cols_to_scale]), index=z_test.index)
    
    return scaler, z_train_scaled, z_validate_scaled, z_test_scaled
    
