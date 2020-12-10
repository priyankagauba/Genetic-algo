# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 10:46:31 2019

@author: u22v03
"""
import pandas as pd
import pickle
import os
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning) 
class distance_attribute:
    def __init__(self,sample_df=None,cycle_duration=None,gear_info=None):
        self.cycle_duration=cycle_duration
        self.df=sample_df
        self.gear_info=gear_info
        
    def calculate_stops_per_km(self):
        self.df['index_']=1

        if self.gear_info=='gear':
            
            output_df=pd.DataFrame([[6,18000]],columns=['Stops_per_xkm','Distance_cycleduration'])
            pickle_path_save=os.path.join(r'output\pickle_files\target_vector',(str(self.cycle_duration)+self.gear_info+'_stops_perkm.pkl'))
            location=open(pickle_path_save,'wb')
            pickle.dump(output_df,location)
            print('stops file created at ',pickle_path_save)
            location.close()
                
        else:
            
            self.df['index_val']=self.df.groupby('date')['index_'].cumsum()
            self.df['tmp_id']=self.df.groupby('date')['index_val'].apply(lambda x:x/(self.cycle_duration*60*10))
            self.df['tmp_id']=self.df.tmp_id.astype('int')
            
            self.df['tmp_mt_id'] = self.df.date+'_'+self.df.tmp_id.map(str)
            final_df=self.df.groupby('tmp_mt_id').agg({'distance':lambda x:x.iloc[-1]-x.iloc[0],'time':'count'})
        
            dist_tmp=round(final_df['distance'].median(),-3)
            #print('dist_tmp',dist_tmp,self.gear_info)
            self.df['distance_id']=self.df.groupby('date')['distance'].transform(lambda x:((x-x.iloc[0]))/dist_tmp).astype('int')
            
            s = self.df.vehicle_speed.eq(0).cumsum().where(self.df.vehicle_speed.ne(0))
            self.df['stops_count'] = self.df.groupby(s).ngroup()+1
            self.df['stops_count']=self.df['stops_count'].replace(to_replace=0, method='ffill')
            
            self.df['stops_per_km']=self.df.groupby(['distance_id','date'])['stops_count'].transform(lambda x:(x.iloc[-1]-x.iloc[0]))
        
            print(self.df['stops_per_km'].describe(),dist_tmp)
            output_df=pd.DataFrame([[7 ,dist_tmp]],columns=['Stops_per_xkm','Distance_cycleduration'])
            pickle_path_save=os.path.join(r'output\pickle_files\target_vector',(str(self.cycle_duration)+'_stops_perkm.pkl'))
            location=open(pickle_path_save,'wb')
            pickle.dump(output_df,location)
            print('stops file created at ',pickle_path_save)
            location.close()
        
        
    def find_stops_per_km(self,sample_df):
        
        s = sample_df.vehicle_speed.eq(0).cumsum().where(sample_df.vehicle_speed.ne(0))
        sample_df['stops_count'] = sample_df.groupby(s).ngroup()+1
        sample_df['stops_count']=sample_df['stops_count'].replace(to_replace=0, method='ffill')
        
        return sample_df['stops_count'].max()
        
        
        
        
    




