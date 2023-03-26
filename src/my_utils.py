import os, sys
import glob
from pathlib import Path

import re, string
import unicodedata

import itertools
from datetime import datetime

import pandas as pd
import numpy as np

from datasets import load_dataset, concatenate_datasets, Dataset

import time
import gc

import logging
import yaml

#-- custom module
from src import constant as my_constant

table = str.maketrans("", "", string.punctuation)



# Get all unicode characters
all_chars = (chr(i) for i in range(sys.maxunicode))

# Get all non printable characters
control_chars = ''.join(c for c in all_chars if unicodedata.category(c) == 'Cc')

# Create regex of above characters
control_char_re = re.compile('[%s]' % re.escape(control_chars))

'''
################################################################################
    - 
'''
def get_configuration():
    try:
        def read_yaml(file_path):
           if os.path.exists(file_path):
            with open(file_path, "r") as f:
               return yaml.safe_load(f)
   
    
        app_config = read_yaml(os.path.join(os.getcwd(), my_constant.yaml_file_name))

        app_setting = app_config.get(my_constant.app)
        search_setting = app_config.get(my_constant.search_setting)

        if app_setting is None or search_setting is None:
            raise Exception("Missing APP/ SEARCH setting key")
        
        working_dir, temp_dir, log_dir = setup_wk_dirs(app_setting)

        if temp_dir is None or working_dir is None or log_dir is None :
            raise Exception('Aborted - Working dir not set!!')
        
        return {
                my_constant.app:app_setting,
                my_constant.search_setting:search_setting,
                'working_dir':working_dir,
                'temp_dir':temp_dir, 
                'log_dir':log_dir

        }
    except Exception as e:
        logging.error(f'YAML file missing or corrupted: {str(e)}')

'''
#####################################
'''
def make_dirs(base_dir):
    if not os.path.exists(base_dir):
       os.makedirs(base_dir)
   
def setup_wk_dirs(app_config):
    '''
        - creates temp, asset, log dirs in getcwd()
    '''
    working_dir = app_config.get(my_constant.asset_dir)

    if working_dir is None:
        working_dir = my_constant.asset

    working_dir = os.path.join(os.getcwd(), working_dir)
    make_dirs(working_dir)

    temp_dir = os.path.join(os.getcwd(), my_constant.temp)
    make_dirs(temp_dir)

    log_dir = os.path.join(os.getcwd(), my_constant.log)
    make_dirs(temp_dir)

    return working_dir, temp_dir, log_dir

def create_file_path_wtimestamp(dir, file_name='tmp_finddoc', extension=my_constant.parquet):
   '''
     - create file_path with time_stamp in filename
     Return str
    '''

   if os.getcwd() in _folder:
       make_dirs(dir)
   else:
       _folder = os.path.join(os.getcwd(), _folder)

   return os.path.join(_folder,  f'{file_name}{str(datetime.now().timestamp())}{extension}')


'''
############################################################################
 - Text cleaning functions
'''
def make_utf8(text):
    '''
     - encode text to utf-8
     Return str
    '''
    return str(text).encode(my_constant.utf_8, 'replace').decode(my_constant.utf_8)

def filter_nonprintable(_text):
    '''
        - removes non-printable characters
        return str   
    '''
    # Use characters of control category
    nonprintable = itertools.chain(range(0x00,0x20),range(0x7f,0xa0))
    
    # Use translate to remove all non-printable characters
    _val = _text.translate({character:None for character in nonprintable})

    res = filter(lambda x: x in string.printable, str(_val))

    return "".join(list(res))

def remove_multiple_spaces(val):
   return re.sub('\s\s+', ' ', val)

# Substitute these characters by empty string in the original string.
def remove_control_chars(s):
    s = control_char_re.sub('', s)
  
    return s

def clean_text(_text):
   '''
        - removes \n\r and convert it to utf-8 encoding
        Return str   
   '''
   
   if _text:
    _text = make_utf8(_text)
    
    if _text:
      _txt = str(_text) 
      #_val = _txt.replace('\n', ' ').replace('\r', '')
      _val = _txt.replace('\\n', ' ').replace('\\r', '').replace('\\t', ' ')
      _val = _val.replace('\\xe2\\x80\\x99s', ' ')

      _val = filter_nonprintable(_val)
      _val = remove_control_chars(_val)
      
      return remove_multiple_spaces(_val)

def remove_punctuation(s):
    return s.translate(table)

def find_all_loc(key, text, max_len=40): 
   '''
    - find all character starting positions of 'key' in 'var', option (int(max_len ) of var
       Return list with start position of each occurence of 'key'
   '''
   pos = [] 
   
   start = 0 
   end = len(text)
   
   while True:
    loc = text.find(key, start, end) 
    
    if loc == -1: 
      break 
    else: 
      pos.append(loc) 
      start = loc + len(key) 
   
   return pos

'''
##############################################################################
    - remove stop words
'''
def remove_stopwords(_text, en_stops):
  if _text:
    _text = convert_list_to_str(_text)
    all_words = str.lower(_text).split()

    words = []

    for word in all_words:
      if word:
        _val = re.sub('[%s]' % re.escape(string.punctuation), '', word) 
        
        if _val not in en_stops:
          words.append(_val)
  
    _words = []

    for _w in words:
      if _w and _w not in _words:
        _words.append(_w) 

    _val = ' '.join(list(set(_words)))

    return re.sub('[%s]' % re.escape(string.punctuation), '', _val)

  return ''

''''
##################################
    - list
'''
def convert_list_to_str(_lst):
  '''
    Converts list -> str
  '''

  if isinstance(_lst, list):
    _val = list(itertools.chain(*_lst))

    _lst = ' '.join([str(_v) for _v in _val])

  return _lst


'''
##############################################################################
    - file manipulation functions

'''
def get_file_encoding(_file):
    '''
        Return file encoding, if can't determine return 'ascii' ->str
    '''
    with open(_file, 'r', errors="surrogateescape") as f:
        _encoding = f.encoding

        return _encoding if _encoding  else "ascii" 
    
def get_rawfile_contents(_file):
    '''
        Return file contents (uses read method)
    '''
    try:
        _encoding = get_file_encoding(_file)

        with open(_file, 'r', encoding=_encoding, errors="surrogateescape") as f:
            return f.read()
    except Exception as e:
        logging.error(f'my_utils.get_file_encoding: {str(e)}')
    

'''
#############################################################################
    - date time function
'''
def get_time_taken(start_tme):
    diff = datetime.now() - start_tme

    return f'{diff.total_seconds():.3f} s'



def load_to_huggingface_dataset(data_path=None, obj=None, start_with='content', max_rec=1000):
    try:
        
        if obj:
           
           if isinstance(obj, dict):
            return Dataset.from_dict(obj, split="train")
        
        if Path(data_path).is_file and os.path.exists(data_path) and data_path.endswith(my_constant.parquet):
           return load_dataset(my_constant.parquet, data_files=data_path, split="train")
  
        _files = glob.glob(data_path.strip(), recursive=False)

        _ds = []

        num_rec = 0
    
        for _f in _files:
            p = Path(_f)
            
            if p.is_dir() or p.name.startswith("."): #ignore hidden paths or dir
               continue

            if  p.name.endswith(my_constant.parquet):
               _d = load_dataset(my_constant.parquet, data_files=_f, split="train")
               _ds.append(_d)
               num_rec = num_rec + len(_d)

            if num_rec > max_rec:
                break
    
        if len(_ds) > 0:
            return concatenate_datasets(_ds)
        
    except Exception as e:
        logging.error(f'my_utils.write_ds_toparquet: {str(e)}')

