#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
# from torchtext.data import utils
import torchvision

import numpy as np
import pandas as pd

from sklearn import metrics
import matplotlib.pyplot as plt

from StaQCDataset import get_dataloader, get_datasets
from transformers import PLBartModel, PLBartTokenizer, TrainingArguments, Trainer, PLBartForConditionalGeneration, get_scheduler

from collections import namedtuple

import json
import os
from transformers import RobertaTokenizer


# In[ ]:


# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")


# ## Set Constants

# #### For Local Training

# In[ ]:


# Set Constants
BATCH_SIZE = 8
EPOCH_SIZE = 5
LEARNING_RATE = .0001
NL_SEQ_LEN = 20
CODE_SEQ_LEN = 150

RESUME_EPOCH = 0                            # Epoch to resume at (0 to start from the beginning)

LOG_STEP = 1                                # Frequency of epoch's for logging
PARAMS_FILE = './training_parameters.json'  # File where input parameters (from cmdline) are stored
PL = 'python'                               # Programming language for fine-tuning the pretrained model
DATA_SIZE = 2000                            # Size of the raw dataset that will be used (batch size * 4 is just for testing)
MODEL_PATH = './checkpoints/plbart_' + PL   # path to store the trained model
MODEL_TYPE = 'plbart'
LOG_FILE = './logs/log_' + MODEL_TYPE + '_' + PL + '.txt'   # Base file path for storing model training 
MODE = 'test'


# #### For rlogin training (COMMENT OUT WHEN RUNNING IPYNB)

# In[ ]:


## Import Enviornment Variables from extenal file 
if os.path.exists(PARAMS_FILE):
    # params = pd.read_csv(PARAMS_FILE)
    with open(PARAMS_FILE) as f:
        params = json.load(f)
    BATCH_SIZE = params['batch_size']
    EPOCH_SIZE = params['epoch_size']
    LEARNING_RATE = params['learning_rate']
    NL_SEQ_LEN = params['nl_seq_len']
    CODE_SEQ_LEN = params['code_seq_len']
    RESUME_EPOCH = params['resume_epoch']
    PL = params['pl_task']
    DATA_SIZE = params['data_size']
    MODEL_TYPE = params['pretrained']
    LOG_STEP = params['log_step']
    LOG_FILE = './logs/log_' + params['pretrained'] + '_' + params['pl_task'] + '.txt'
    MODE = params['mode']
    
    # Check if files exist       
    MODEL_PATH = './checkpoints/' + params['pretrained'] + '_' + params['pl_task']
    if not os.path.isdir(MODEL_PATH):
        os.mkdir(MODEL_PATH)


# In[ ]:


print("RUNNING: {} with pl={}, mode={}, batch_size={}, epoch_size={}, learning_rate={}, nl_seq_len={}, code_seq_len={}, data_size={}, log_step={}, log_file={}, model_dir={} resume_epoch={}".format(MODEL_TYPE, PL, MODE, BATCH_SIZE, EPOCH_SIZE, LEARNING_RATE, NL_SEQ_LEN, CODE_SEQ_LEN, DATA_SIZE, LOG_STEP, LOG_FILE, MODEL_PATH, RESUME_EPOCH))


# ## Create Dataloaders

# In[ ]:


def tokenize_func(data, tokenizer=None, seq_len=(20, 150)):
    # Dissect data
    code, nl = list(data['code']), list(data['nl'])
    nl_len, code_len = seq_len
    
    # Tokenize
    if tokenizer == None:
        tokenizer = PLBartTokenizer.from_pretrained("uclanlp/plbart-base")
        
    inputs = tokenizer(nl, return_tensors='pt', max_length=nl_len, padding="max_length", truncation=True)
    labels = tokenizer(code, return_tensors="pt", max_length=code_len, padding="max_length", truncation=True)
    # with tokenizer.as_target_tokenizer():
    #     labels = tokenizer(code, return_tensors="pt", max_length=code_len, padding="max_length", truncation=True)
    
    InputFeatures = namedtuple("InputFeatures", "input_tokens input_ids input_mask target_tokens target_ids target_mask")
    
    # Masked input Ids
    # mask_label_ids = torch.where(labels.attention_mask == 1, labels.input_ids, torch.tensor(-100))
    
    data = InputFeatures(None, inputs.input_ids, inputs.attention_mask, None, labels.input_ids, labels.attention_mask)
    
    return data


# In[ ]:


# Training DataLoader
data_path = './staqc_data/' + PL 
train_loader, test_loader, val_loader = get_dataloader(data_path, tokenize_func, 
                                                       batch_size=BATCH_SIZE, 
                                                       data_size=DATA_SIZE)


# ## Create or Load the PLBART Model

# In[ ]:


torch.cuda.empty_cache()

# Load trained and saved model if needed
model = PLBartForConditionalGeneration.from_pretrained("uclanlp/plbart-base")
if RESUME_EPOCH > 0:
    saved_model_path = '{}/Epoch{}.pkl'.format(MODEL_PATH, str(RESUME_EPOCH))
    if os.path.exists(saved_model_path):
        model.load_state_dict(torch.load(saved_model_path))
    else:
        print("WARNING: {} saved model does not exist! Training {} model from the epoch 0.".format(saved_model_path, MODEL_TYPE))
model.to(device)


# ## Train the Model

# In[ ]:


if MODE == 'train':  
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Create a training scheduler for learning rate stepping
    num_training_steps = EPOCH_SIZE * len(train_loader)
    lr_scheduler = get_scheduler(name='linear', optimizer=optimizer, num_training_steps=num_training_steps, num_warmup_steps=0)

    # Log training
    train_loss = list()
    val_loss = list()

    # open log file
    if RESUME_EPOCH > 0:
        log_f = open(LOG_FILE, 'a')
    else:
        log_f = open(LOG_FILE, 'w')

    # Training Loop
    for epoch in range(RESUME_EPOCH, EPOCH_SIZE):

        # Initialize batch loss vars
        model.train()
        running_loss = list()
        for batch in train_loader:

            # Dissect batch
            _, input_ids, _, _, target_ids, _ = batch
            input_ids = input_ids.to(device)
            # input_mask = input_mask.to(device)
            target_ids = target_ids.to(device)
            # target_mask = target_mask.to(device)

            outputs = model(input_ids = input_ids, labels=target_ids)

            # Calculate loss
            loss = outputs.loss
            running_loss.append(loss.item())
            loss.backward()

            # Update Gradients
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        curr_train_loss = np.array(running_loss).mean()
        train_loss.append(curr_train_loss)

        # Validation
        model.eval()
        running_loss = list()
        for batch in val_loader:
            # Dissect batch
            _, input_ids, _, _, target_ids, _ = batch
            input_ids = input_ids.to(device)
            # input_mask = input_mask.to(device)
            target_ids = target_ids.to(device)
            # target_mask = target_mask.to(device)

            # Forward Pass through the model
            outputs = model(input_ids = input_ids, labels=target_ids)

            # Calculate loss
            loss = outputs.loss
            running_loss.append(loss.item())

        curr_val_loss = np.array(running_loss).mean()
        val_loss.append(curr_val_loss)

        # Log loss for epoch
        if epoch % LOG_STEP == 0:
            loss_log = "Epoch: [{}/{}], Training Loss: {:02.6f}, Validation Loss: {:02.6f}".format(epoch + 1, EPOCH_SIZE, curr_train_loss, curr_val_loss)
            log_f.write(loss_log + '\n')
            print(loss_log)

        # Save checkpoint
        torch.save(model.state_dict(), '{}/Epoch{}.pkl'.format(MODEL_PATH, str(epoch+1)))

        # Quit if the the validation loss passes training loss for more than 2 epochs
        val_pass = True
        for t, v in zip(train_loss[-2:], val_loss[-2:]):
            val_pass = val_pass and (v > t)

        if val_pass and len(train_loss) > 3:
            print("EARLY STOP: Validation loss was greater than training loss for the last 3 epochs!")
            break

    log_f.close()


# ### Graph Training vs Validation Loss

# In[ ]:


if MODE == 'train':
    train_loss, val_loss = list(), list()
    for line in open(LOG_FILE):
        split_line = line.split(' ')
        train_loss.append(float(split_line[4].strip().replace(',', '')))  # Pull training loss from logs (from rlogin)
        val_loss.append(float(split_line[7].strip()))    # pull validation loss from logs

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(MODEL_TYPE.upper() + " Loss vs Epoch")
    plt.plot(train_loss,'k')
    plt.plot(val_loss,'y')
    plt.legend(["Training Loss","Validation Loss"])
    plt.savefig(MODEL_TYPE.upper() + '_' + PL.upper() + '_Training_Validation_Loss.png')
    plt.show()


# ### Testing

# In[ ]:


if MODE == 'test':
    test_base = "./CodeBLEU/test_files"
    if not os.path.isdir(test_base):
        os.mkdir(test_base)
    test_base = os.path.join(test_base, MODEL_TYPE + '_' + PL)
    if not os.path.isdir(test_base):
        os.mkdir(test_base)
     
    inp_file = open(test_base + '/input.txt', 'w')
    hyp_file = open(test_base + '/hypothesis.txt', 'w')
    ref_file = open(test_base + '/reference.txt', 'w')
        
    print("Testing.....")
    batch_num = 0
    inputs, hypothesis, reference = [], [], []
    model.eval()
    for batch in test_loader:
        # Dissect batch
        _, input_ids, _, _, target_ids, _ = batch
        input_ids = input_ids.to(device)
        
        # Generate ids with input batch
        output_ids = model.generate(input_ids = input_ids)
        
        # Save results for CodeBleu Later on
        inputs.extend(input_ids)
        hypothesis.extend(output_ids)
        reference.extend(target_ids)
        
        batch_num += 1
        if batch_num % 100 == 0:
            print("Batch {} done!".format(batch_num))
        
    # Covert and save decoded values 
    tokenizer = PLBartTokenizer.from_pretrained("uclanlp/plbart-base")
    decoded_input = tokenizer.batch_decode(inputs, skip_special_tokens=True)
    decoded_hyp = tokenizer.batch_decode(hypothesis, skip_special_tokens=True)
    decoded_refs = tokenizer.batch_decode(reference, skip_special_tokens=True)
    
    for i, h, r in zip(decoded_input, decoded_hyp, decoded_refs):
        inp_file.write(i + '\n')
        hyp_file.write(h + '\n')
        ref_file.write(r + '\n')
            
inp_file.close()
hyp_file.close()
ref_file.close()
                


# In[ ]:




