import os, sys
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset


from utils.sequence_indexing import Sequence_indexing
from utils.dataset import TransformerDataset

class Data_processing(object):
    
    def min_max_norm(self, ts, min_, max_):
        ts = np.array(ts)
        return (ts - min_)/(max_ - min_)


    def identifyLinearTrend(self,ixValues, iyValues):
    
            counter = 0;
            denominator = 0;
            avgX = np.mean(ixValues)
            avgY = np.mean(iyValues)
    
            nrValues = len(ixValues);
    
            for i in range(nrValues):
                tmpXiXavg = ixValues[i] - avgX;
                tmpYiYavg = iyValues[i] - avgY;
                
                counter += tmpXiXavg * tmpYiYavg;
                denominator += tmpXiXavg * tmpXiXavg;
     
            if (denominator == 0): 
              slope = sys.float_info.max
            else:
              slope = counter / denominator;
    
            intercept = avgY - slope * avgX;
        
            return  slope, intercept 
        
    def detrend(self,x, y):
            x = np.array(x)
            y = np.array(y)
            slope, intercept = self.identifyLinearTrend(x, y)
            #print(slope, intercept)
            #slope_intercept = np.polyfit(x,y,1)
            #trend = slope_intercept[0]*x + slope_intercept[1]
            trend = slope*x + intercept
            return y - trend, trend
    
    def data_loader(self):
        df = pd.read_csv("logs/energydata_complete_formatted.csv")
        keys = list(df.keys())
        sub_df = df[keys[:2]]
        keys = list(sub_df.keys())
        timestamp = np.array(sub_df[keys[0]])
        values = np.array(sub_df[keys[1]])
        
        
        timestamp= self.min_max_norm(timestamp, np.min(timestamp), np.max(timestamp))
        values= self.min_max_norm(values, np.min(values), np.max(values))
        
        values, trend = self.detrend(timestamp, values) 
        
        return timestamp, values
    
    
    def data_formating(self, dataset, window_size, step_size, enc_seq_len, dec_seq_len,\
                    output_sequence_length, batch_first, batch_size  ):
        
        seq_ind = Sequence_indexing()
        # Make list of (start_idx, end_idx) pairs that are used to slice the time series sequence into chunkc. 
        # Should be training data indices only
        indices = seq_ind.get_indices_entire_sequence(
            data=dataset, 
            window_size=window_size, 
            step_size=step_size)
        #print("indices", indices)
        
        # Making instance of custom dataset class
        data = TransformerDataset(
            data=dataset,
            indices=indices,
            enc_seq_len=enc_seq_len,
            dec_seq_len=dec_seq_len,
            target_seq_len=output_sequence_length
            )
        
        #print("data", next(enumerate(data)))
        
        # Making dataloader
        data = DataLoader(data, batch_size)
        #print("training_data.shape", np.shape(training_data.dataset[0]), np.shape(training_data.dataset[1]), np.shape(training_data.dataset[2]))
        i, batch = next(enumerate(data))
        
        src, trg, trg_y = batch
        print(src.shape, trg.shape, trg_y.shape)
    
        #print("src", src)
        #print("trg", trg)
        #print("trg_y", trg_y)
       
        if batch_first == False:
    
            shape_before = src.shape
            src = src.permute(1, 0, 2)
            print("src shape changed from {} to {}".format(shape_before, src.shape))
        
            shape_before = trg.shape
            trg = trg.permute(1, 0, 2)
            print("src shape changed from {} to {}".format(shape_before, src.shape))
            
        return data
            