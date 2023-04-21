
"""
Code from https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
"""
import os, sys
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn 
from torch import nn, Tensor
from typing import Tuple
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, TensorDataset


from transformer.TimeSeriesTransformer import TimeSeriesTransformer
from utils.data_processing import Data_processing
from utils.sequence_indexing import Sequence_indexing

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)



def run_training(model, epochs, training_dataloader, valid_dataloader, output_sequence_length, enc_seq_len, \
                 criterion, optimizer ):
    
    seq_ind = Sequence_indexing()
    
    # Iterate over all epochs
    for epoch in range(epochs):
        total_loss = 0.
        # Iterate over all (x,y) pairs in training dataloader
        for i, batch in enumerate(training_dataloader):
            
            src, trg, tgt_y = batch
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # Make src mask for decoder with size:
            # [batch_size*n_heads, output_sequence_length, enc_seq_len]
            src_mask = seq_ind.generate_square_subsequent_mask(
                dim1=output_sequence_length,
                dim2=enc_seq_len
                )
            
            # Make tgt mask for decoder with size:
            # [batch_size*n_heads, output_sequence_length, output_sequence_length]
            tgt_mask = seq_ind.generate_square_subsequent_mask( 
                dim1=output_sequence_length,
                dim2=output_sequence_length
                )
            #print("src.shape", src.shape, "src mask", src_mask.shape, "trg ", trg.shape, "tgt_mask ", tgt_mask.shape)
            
            # Make forecasts
            prediction = model(
                        src=src.float().to(device=device),
                        tgt=trg.float().to(device=device),
                        src_mask=src_mask.float().to(device=device),
                        tgt_mask=tgt_mask.float().to(device=device)
                        )
    
            #print("pred shape ",prediction.shape,"tgt_y_shape", tgt_y.shape,"pred permuted shape", (prediction.permute(1, 2, 0)).shape )
            
            #prediction = prediction.permute(1, 2, 0)
            
            # Compute and backprop loss
            loss = criterion(tgt_y.float(), prediction)
            total_loss += loss.item()
            loss.backward()
            
            #print("epoch: ", epoch, "loss: ", loss.item())
    
            # Take optimizer step
            optimizer.step()

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss/(i+1)
                }, "logs/model.pt")
            

        #print("epoch: ", epoch, "loss: ", total_loss/i)
        
        # Iterate over all (x,y) pairs in validation dataloader
        model.eval()
        validation_loss = 0
        with torch.no_grad():
        
            for i, (src, trg, tgt_y) in enumerate(valid_dataloader):
    
                prediction = model(
                    src=src.float().to(device=device),
                    tgt=trg.float().to(device=device),
                    )
    
                loss = criterion(tgt_y, prediction)
                validation_loss +=  loss.item()
        print("Epoch: %d, loss: %1.5f  valid loss:  %1.5f " %(epoch, total_loss/(i+1), validation_loss/(i+1)))
    return model
    
def get_prediction(model, test_dataloader):
    model.eval()
    
    results = []
    targets = []
    with torch.no_grad():
    
        for i, (src, trg, tgt_y) in enumerate(test_dataloader):

            prediction = model(
                src=src.float().to(device=device),
                tgt=trg.float().to(device=device),
                )

            results.append((prediction.cpu().detach().numpy()).reshape(-1))
            targets.append((tgt_y.cpu().detach().numpy()).reshape(-1))
    print("results", results)
    print("targets", targets)
    return results, targets

def main():
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    

    dim_val = 512 # This can be any value divisible by n_heads. 512 is used in the original transformer paper.
    n_heads = 8 # The number of attention heads (aka parallel attention layers). dim_val must be divisible by this number
    n_decoder_layers = 4 # Number of times the decoder layer is stacked in the decoder
    n_encoder_layers = 4 # Number of times the encoder layer is stacked in the encoder
    input_size = 2 # The number of input variables. 1 if univariate forecasting.
    dec_seq_len = 168 # length of input given to decoder. Can have any integer value.
    enc_seq_len = 168 # length of input given to encoder. Can have any integer value. # supposing you want the model to base its forecasts on the previous 7 days of data = 168
    output_sequence_length = 1 # Length of the target sequence, i.e. how many time steps should your forecast cover # supposing you're forecasting 48 hours ahead = 48
    window_size = enc_seq_len + output_sequence_length # used to slice data into sub-sequences
    step_size = 1 # Step size, i.e. how many time steps does the moving window move at each step
    max_seq_len = enc_seq_len # What's the longest sequence the model will encounter? Used to make the positional encoder
    batch_first = True
    batch_size = 20

    
    #training_data = torch.arange(0,10080,1, dtype=torch.float32)
    #test_data = torch.arange(0,1440,1, dtype=torch.float32)
    
    dp = Data_processing()
    timestamp, values = dp.data_loader()
    training_set_size = 5000
    timestamp = torch.tensor(np.array(timestamp), dtype=torch.float,  device=device)
    values = torch.tensor(np.array(values), dtype=torch.float,  device=device)
    
    training_ts = timestamp[:training_set_size]
    training_val = values[:training_set_size]
    training_data = torch.vstack((training_ts,training_val)).T
    training_data = torch.tensor(training_data, dtype=torch.float)
    
    valid_set_size = 500
    valid_ts = timestamp[training_set_size:(training_set_size + valid_set_size)]
    valid_val = values[training_set_size:(training_set_size + valid_set_size)]
    valid_data = torch.vstack([valid_ts,valid_val]).T
    valid_data = torch.tensor(valid_data, dtype=torch.float)
    
    test_set_size = 500
    N = training_set_size +  valid_set_size+ test_set_size
    test_ts = timestamp[N - test_set_size : N]
    test_val = values[N - test_set_size : N]
    test_data = torch.vstack([test_ts,test_val]).T
    test_data = torch.tensor(test_data, dtype=torch.float)
    
    
    training_dataloader = dp.data_formating( training_data, window_size, step_size, enc_seq_len, dec_seq_len,\
                    output_sequence_length, batch_first, batch_size  )
    
    valid_dataloader = dp.data_formating( valid_data, window_size, step_size, enc_seq_len, dec_seq_len,\
                    output_sequence_length, batch_first, batch_size  )
    
    test_dataloader = dp.data_formating( test_data, window_size, step_size, enc_seq_len, dec_seq_len,\
                    output_sequence_length, batch_first, batch_size  )
    
    
    model = TimeSeriesTransformer(
        input_size= input_size,
        dec_seq_len=enc_seq_len,
        batch_first=batch_first,
        num_predicted_features= input_size
        )

    epochs = 1000
    forecast_window = 1 # supposing you're forecasting 48 hours ahead

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    """
    if os.path.isfile("logs/model.pt"):
        checkpoint = torch.load("logs/model.pt", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        results, targets = get_prediction(model, test_dataloader)
    
        results = np.array(results).reshape(-1)
        targets = np.array(targets).reshape(-1)
        print(results, targets)
    """    
    model = run_training(model, epochs, training_dataloader, valid_dataloader, output_sequence_length, enc_seq_len, \
                 criterion, optimizer )
    
    results, targets = get_prediction(model, test_dataloader)
    
    #results = np.array(results).reshape(-1)
    #targets = np.array(targets).reshape(-1)
    #print(results, targets)
    
   
        
if __name__ == '__main__':
    main()
    
    
