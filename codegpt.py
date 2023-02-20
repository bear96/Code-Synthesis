#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torchvision

import numpy as np
import pandas as pd

from sklearn import metrics
import matplotlib.pyplot as plt

from collections import namedtuple

import json
import os

import pandas as pd
import numpy as np 
from transformers import RobertaTokenizer, AutoModelForCausalLM,GPT2Tokenizer,GPT2LMHeadModel,GPT2Config
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader,TensorDataset
from matplotlib import pyplot as plt


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
MODEL_PATH = './checkpoints/codegpt_' + PL   # path to store the trained model
MODEL_TYPE = 'codegpt'
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


# ## Concode Dataset

# In[ ]:


class concodeDataset(Dataset):
    def __init__(self, tokenizer, data, block_size=150, mode='train'):

            self.block_size = block_size
            self.mode = mode
            self.inputs = []
            self.token_labels = []

            datas = data

            length = len(datas)

            for idx in range(len(datas)):
                x = datas.iloc[idx]
                code = tokenizer.encode(x["Code"])
                nl = tokenizer.encode(x["NL"])

                input_ids, input_labels = self.pad_and_get_mask(code, nl, tokenizer)
                self.inputs.append(input_ids)
                self.token_labels.append(input_labels)


    def pad_and_get_mask(self, code, nl, tokenizer):
        if self.mode == 'test':
            code = []
        while (len(code) + len(nl) + 2 > self.block_size):
            if (len(code) > len(nl)):
                code = code[:-1]
            else:
                nl = nl[:-1]
        if self.mode == 'train':
            inputs = nl + [tokenizer.bos_token_id] + code + [tokenizer.eos_token_id]
            labels = [1] * len(nl) + [2] * (len(code)+1) + [0]
        else:
            inputs = nl + [tokenizer.bos_token_id]
            labels = [1] * len(nl) + [2]
            return inputs, labels
        assert len(inputs) <= self.block_size
        pad_len = self.block_size - len(inputs)
        inputs += [tokenizer.pad_token_id] * pad_len
        labels += [0] * pad_len
        assert len(inputs) == len(labels)
        return inputs, labels


    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return torch.tensor(self.inputs[item]), torch.tensor(self.token_labels[item])


# ## Create or Load the PLBART Model

# In[ ]:


torch.cuda.empty_cache()

# Load trained and saved model if needed
CodeGPT = GPT2LMHeadModel.from_pretrained("microsoft/CodeGPT-small-py-adaptedGPT2")


# ## Create Dataloaders

# In[ ]:


tokenizer = GPT2Tokenizer.from_pretrained("microsoft/CodeGPT-small-py-adaptedGPT2",do_lower_case=True, bos_token='<s>', eos_token='</s>', pad_token='<pad>', unk_token='<|UNKNOWN|>', sep_token='concode_elem_sep')
config = GPT2Config.from_pretrained("microsoft/CodeGPT-small-py-adaptedGPT2")

config.max_length = 150

CodeGPT.resize_token_embeddings(len(tokenizer))

CodeGPT.config.bos_token_id = tokenizer.bos_token_id
CodeGPT.config.eos_token_id = tokenizer.eos_token_id
CodeGPT.config.pad_token_id = tokenizer.pad_token_id
CodeGPT = CodeGPT.to(device)

if MODE == 'train':
    data = pd.read_csv('./staqc_data/' + PL + '/train.csv')
    val_data = pd.read_csv('./staqc_data/' + PL + '/val.csv')

    dataset = concodeDataset(tokenizer, data, block_size=150)
    val_dataset = concodeDataset(tokenizer, val_data,mode='train', block_size=150)

    train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, drop_last=True,shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=True,shuffle=False)
else:
    test_data = pd.read_csv('./staqc_data/' + PL + '/test.csv')
    test_dataset = concodeDataset(tokenizer, test_data, mode='test', block_size=20)
    test_loader = DataLoader(test_dataset, batch_size=1, drop_last=True,shuffle=False)


# In[ ]:


if RESUME_EPOCH > 0:
    saved_model_path = '{}/Epoch{}.pkl'.format(MODEL_PATH, str(RESUME_EPOCH))
    if os.path.exists(saved_model_path):
        CodeGPT.load_state_dict(torch.load(saved_model_path))
    else:
        print("WARNING: {} saved model does not exist! Training {} model from the epoch 0.".format(saved_model_path, MODEL_TYPE))
CodeGPT.to(device)


# ## Train the Model

# In[ ]:


torch.cuda.empty_cache()
if MODE == 'train':
    train_loss_graph = []
    val_loss_graph = []
    for i in range(EPOCH_SIZE):
        tr_loss = 0.0
        eval_loss = 0.0
        for batch,token_labels in train_dataloader:
            optimizer.zero_grad()
            token_labels = token_labels.to(device)
            attn_mask = torch.tensor(token_labels.clone().detach() != 0, dtype=torch.uint8)
            loss_mask = torch.tensor(token_labels.clone().detach() == 2, dtype=torch.uint8)
            attn_mask = attn_mask.to(device)
            batch = batch.to(device)
            CodeGPT.train()
            out = CodeGPT(batch,attention_mask=attn_mask)
            logits = out.logits
            labels = batch
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            flatten_shift_loss_mask = loss_mask[..., :-1].contiguous().view(-1)
            ids = torch.nonzero(flatten_shift_loss_mask).view(-1)
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1))[ids], shift_labels.view(-1)[ids])
            loss.backward()
            optimizer.step()
            tr_loss+= loss.item()

        for batch,token_labels in val_dataloader:
            CodeGPT.eval()
            with torch.no_grad():
                token_labels = token_labels.to(device)
                attn_mask = torch.tensor(token_labels.clone().detach() != 0, dtype=torch.uint8)
                loss_mask = torch.tensor(token_labels.clone().detach() == 2, dtype=torch.uint8)
                attn_mask = attn_mask.to(device)
                batch = batch.to(device)
                out = CodeGPT(batch,attention_mask=attn_mask)
                logits = out.logits
                labels = batch
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                flatten_shift_loss_mask = loss_mask[..., :-1].contiguous().view(-1)
                ids = torch.nonzero(flatten_shift_loss_mask).view(-1)
                loss = criterion(shift_logits.view(-1, shift_logits.size(-1))[ids], shift_labels.view(-1)[ids])
                eval_loss += loss.item()
                #print(eval_loss)


        tr_loss = tr_loss/len(train_dataloader)
        eval_loss = eval_loss/len(val_dataloader)
        train_loss_graph.append(tr_loss)
        val_loss_graph.append(eval_loss)
        print("Epoch: {} Train Loss: {} Val Loss: {}".format(i+1,tr_loss,eval_loss))
        # Save checkpoint
        torch.save(model.state_dict(), '{}/Epoch{}.pkl'.format(MODEL_PATH, str(i+1))) 


# ### Graph Training vs Validation Loss

# In[ ]:


if MODE == 'train':
    plt.plot(train_loss_graph,'k')
    plt.plot(val_loss_graph,'y')
    plt.legend(["Training Loss","Validation Loss"])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.savefig(MODEL_TYPE.upper() + '_' + PL.upper() + '_Training_Validation_Loss.png')

    losses = pd.DataFrame({'Tr_Loss': train_loss_graph, 'Val_Loss': val_loss_graph})

    losses.to_csv('Training_loss_py_GPT.csv')


# ### Testing

# In[ ]:


if MODE == 'test':

    print("Testing.....")
    batch_num = 0
    inputs, hypothesis, reference = [], [], []
    CodeGPT.eval()
    for batch, token_labels in test_loader:
        step+=1
        if step >= 2000:
            step=0
            break
        inputs = batch.to(device)
        with torch.no_grad():
            outputs = CodeGPT.generate(inputs, max_length=150, num_beams=10, temperature=0.7, early_stopping=False, top_k=70,                                    bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
            generation = tokenizer.decode(outputs[0])[len(tokenizer.decode(inputs[0])):]
            preds.append(generation.rstrip("<pad>"))

    d = pd.DataFrame({'predicted_codes':preds})
    d.to_csv("SQL_predictions_CodeGPT.csv")
                

