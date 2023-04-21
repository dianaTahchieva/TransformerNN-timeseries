# transformerNN



## Data processing

The timeseries are normalised using the min-max normalisation scheem. Afterwards, they are detrended by subtacting the linear regression prediction of the timestamps.

The data is then formatted according to https://towardsdatascience.com/how-to-make-a-pytorch-transformer-for-time-series-forecasting-69e073d4061e ,  which is:
 - [ ] [generate a tuple of start-end indeces per sequence length, where each tuple is (start_idx, end_idx) of a sub-sequence]
 - [ ] [generate encoder-, decoder- input and target label](For each sub-sequence of data[start_idx : end_idx] generate encoder input (src) as src = sequence[:enc_seq_len],
decoder input (trg) as trg = sequence[enc_seq_len-1:len(sequence)-1] and the target label as trg_y = sequence[-number_of_points_to_predict:].)

Torch.utils.data.DataLoader is used for batching.




 
