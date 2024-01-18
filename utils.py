from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class  RenameColumns (BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, df):
        return self
    
    def transform(self, df):
        df.rename(columns= {'data' : 'ds', 'preco_petroleo_brent': 'y'}, inplace=True)
        return df


class CastToFloat(BaseEstimator, TransformerMixin):
    def __init__(self, ft_to_cast='y'):        
        self.ft_to_cast = ft_to_cast


    def fit(self, df):
        return self
    
    def transform(self, df):
        if set([self.ft_to_cast]).issubset(df.columns):
            try:
                df[self.ft_to_cast] = df[self.ft_to_cast].str.replace(',','.').astype(float)
            except ValueError:                
                pass
            
        return df


class CastToDatetime(BaseEstimator, TransformerMixin):
    def __init__(self, ft_to_cast='ds'):        
        self.ft_to_cast = ft_to_cast


    def fit(self, df):
        return self
    
    def transform(self, df):
        if set([self.ft_to_cast]).issubset(df.columns):
            try:                
                df[self.ft_to_cast] = pd.to_datetime(df[self.ft_to_cast])
            except ValueError:                          
                pass
        
        return df


class FillMissingData (BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, df):
        return self
    
    def transform(self, df):
        date_range = pd.date_range(start=df['ds'].min(), end=df['ds'].max(), freq='D')
        df_date_range = pd.DataFrame({'ds': date_range})
        df_completo = pd.merge(df_date_range, df, on='ds', how='left')
        df_completo.sort_values(by='ds', ascending=False, inplace=True)
        df_completo['y'] = df_completo['y'].fillna(method='bfill')
        df_completo.reset_index(drop=True, inplace=True)       

        return df_completo


class AddColumn (BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, df):
        return self
    
    def transform(self, df):       
        df['unique_id'] = 'value'       

        return df