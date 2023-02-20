#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import torchvision.transforms as transforms
import torch.utils.data as data
import torch


# In[2]:


class StaQCDataset(data.Dataset):
    """Custom Dataset for pseudocode - code pairs"""
    def __init__(self, root, tokenizer_func, tokenizer=None, max_seq_len=(20,150), data_size=None, data_cleaner=None):

        if data_cleaner is None:  # If the dataset is a DeepPseudo dataset
            # Load Data
            data = pd.read_csv(root)
            
            # Preprocess and clean the dataset
            data.rename(columns={'Code':'code', 'NL':'nl'}, inplace=True)
            
            if data_size != None:
                data = data[:data_size]
            
            data['code'] = [line.strip() for line in data['code']]
            data['nl'] = [line.strip().lower() for line in data['nl']]
            
        else:                     
        # If the dataset is not from StacQC a custom cleaning function 
        # could be used to create similar format
            data = data_cleaner(root)
          
        # Tokenize the cleaned dataset
        self.data = data
        
        self.samples = tokenizer_func(data, tokenizer, max_seq_len)
        self.input_ids = self.samples.input_ids
        self.input_mask = self.samples.input_mask
        self.target_ids = self.samples.target_ids
        self.target_mask = self.samples.target_mask
        
        
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        """ Returns one elemet of the dataset """
            
        return self.input_ids[index], self.input_mask[index], self.target_ids[index], self.target_mask[index]


# In[ ]:


def get_datasets(root, tokenizer_func, data_size, train, test, val, max_seq_len, tokenizer=None, data_cleaner=None):
    # Ensure inputs are in the ranges expected
    #if data_size > 85000:
    #   raise ValueError("data_size must be below the maximum size of dataset: 85K.")
    if train + test + val != 1:
        raise ValueError("Sum of train, test and val fractions must add to 1!")
        
    # Create Train, Test, Validation Splits when smaller data is requested for testing
    train_size = int(train*data_size) if data_size != None else None
    test_size = int(data_size * test) if data_size != None else None
    val_size = (data_size - train_size - test_size) if data_size != None else None
    
    train_data = StaQCDataset(os.path.join(root, 'train.csv'), tokenizer_func, tokenizer, max_seq_len, train_size, data_cleaner)
    test_data = StaQCDataset(os.path.join(root, 'test.csv'), tokenizer_func, tokenizer, max_seq_len, test_size, data_cleaner)
    val_data = StaQCDataset(os.path.join(root, 'val.csv'), tokenizer_func, tokenizer, max_seq_len, val_size, data_cleaner)
    
#     if os.path.isdir("./staqc_data/train") and os.path.isdir(".staqc_data/test"):
#         pass
#     else:
#         # Initialize custom dataset
#         dataset = StaQCDataset(root, tokenizer_func, tokenizer, max_seq_len, data_size, data_cleaner)

#         data_size = len(dataset)

#         # Create Train, Test, Validation Splits
#         train_size = int(train*data_size)
#         test_size = int(data_size * test)
#         val_size = data_size - train_size - test_size
#         train_data, test_data, val_data = data.random_split(dataset, [train_size, test_size, val_size])
               
    print("Train Size:{}\nTest Size:{}\nValidation Size:{}".format(len(train_data), len(test_data), len(val_data))) 
    
    return train_data, test_data, val_data


# In[4]:


def get_dataloader(root, tokenizer_func, 
                   batch_size=32, 
                   max_seq_len=(20,150), 
                   data_size=None, 
                   train=.70, test=.15, val=.15, 
                   shuffle=True, num_workers=0, 
                   tokenizer=None, data_cleaner=None):
    """ Creates a train, test, and val dataloader with the collate function"""
    
    train_data, test_data, val_data = get_datasets(root, tokenizer_func, data_size, train, test, val, max_seq_len, tokenizer, data_cleaner)
    
    # Create dataloaders for each dataset
    train_loader = data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = data.DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader, val_loader


# In[ ]:




