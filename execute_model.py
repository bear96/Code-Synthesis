#!/usr/bin/env python
# coding: utf-8

# In[3]:


import argparse
import os
import json
import subprocess


# In[1]:


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8, help='Batch Size')
    parser.add_argument('--epoch_size', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=.0001, help='Training learning rate')
    parser.add_argument('--nl_seq_len', type=int, default=20, help='The size of the nl input to model')
    parser.add_argument('--code_seq_len', type=int, default=150, help='The size of the code input to model')
    parser.add_argument('--pretrained', type=str, default='plbart', help='The pretrained model to train [plbart, codet5]')
    parser.add_argument('--log_step', type=int, default=1, help='Number of epochs in between in each logging')
    parser.add_argument('--pl_task', type=str, default='python', help="The Programming Language to train [--pretrained] model on. ['python', 'sql']")
    parser.add_argument('--data_size', type=int, default=None, help="The size of the raw dataset that will be used.")
    parser.add_argument('--resume_epoch', type=int, default=0, help="The epoch at which training resumes.")
    parser.add_argument('--mode', type=str, default='train', help='training or testing mode.')
    args = parser.parse_args()
    
    # Convert arguments to JSON for storage
    params = json.dumps(vars(args))
    with open('training_parameters.json', 'w') as f:
        f.write(params)
    
    # Convert all the ipynb files to py
    os.system("jupyter nbconvert --to script *.ipynb")
    print("Converted all '.ipynb' files to '.py'.")
    
    models = ['plbart', 'codet5', 'codebert', 'codegpt']
    if args.pretrained in models:
        if os.path.exists(args.pretrained + '.py'):
            os.system('python3 ' + args.pretrained + '.py')
        else:
            print("ERROR: Pretrained {} model does not exist!".format(args.pretrained))
    else:
        print("ERROR: Model has to be one of {}.".format(models))




